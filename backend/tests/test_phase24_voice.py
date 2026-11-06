"""Phase 24 - two-way voice conversation (WebSocket + VAD + STT + TTS).

What is covered:

* ``EnergyVAD`` turn detection in isolation (silence, speech, hangover,
  echo margin, floor retention);
* WAV chunk splitting (browser-bound mini-WAVs stay playable);
* the ``/ws/voice`` protocol end to end with **fake** STT/TTS and the
  faked LLM: start/ready, transcript -> thinking -> reply -> audio ->
  listening, barge-in interruption, pause/resume, mute, dedupe, empty
  transcripts (no model call), STT/TTS failure fallbacks;
* safety: a crisis utterance through the voice path hits the *same*
  pre-generation gate (``source == "pre_blocked"``, LLM never called);
* persistence: authenticated voice turns write exactly one user row and
  one assistant row per turn - no duplicates - into the existing store,
  on the existing session;
* auth: missing/invalid token closes with 4401; a foreign session id is
  rejected exactly like ``POST /api/chat``.

No network, no model weights: Whisper/Piper are replaced by injected
fakes (``set_transcriber`` / ``set_synthesizer``).
"""

from __future__ import annotations

import base64
import json
import os
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.voice.stt import STTError, set_transcriber
from app.voice.tts import TTSError, set_synthesizer, split_wav_chunks, wav_bytes
from app.voice.vad import EnergyVAD
from tests.fakes import BLOCKER, BENIGN, FakeLLM, api_client, auth_env

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 480  # 30 ms @ 16 kHz
FRAME_BYTES = FRAME_SAMPLES * 2


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def silence_frame() -> bytes:
    return b"\x00" * FRAME_BYTES


def tone_frame(amp: int = 9000, freq: float = 440.0) -> bytes:
    t = np.arange(FRAME_SAMPLES)
    sig = (amp * np.sin(2 * np.pi * freq * t / SAMPLE_RATE)).astype(np.int16)
    return sig.tobytes()


def audio_msg(pcm: bytes) -> str:
    return json.dumps(
        {"type": "audio", "data": base64.b64encode(pcm).decode("ascii")}
    )


def utterance() -> bytes:
    """2 frames of silence + 8 speech frames + 15 hangover frames."""
    return silence_frame() * 2 + tone_frame() * 8 + silence_frame() * 15


class ScriptedSTT:
    """Fake transcriber: pops scripted texts, then returns ''."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls = 0

    def __call__(self, audio: np.ndarray, language: str | None = None) -> str:
        self.calls += 1
        if not self.texts:
            return ""
        return self.texts.pop(0)


def fake_tts(text: str) -> bytes:
    """2 seconds of tone -> 5 chunks at the default 400 ms framing."""
    t = np.arange(SAMPLE_RATE * 2)
    pcm = (6000 * np.sin(2 * np.pi * 220 * t / SAMPLE_RATE)).astype(np.int16)
    return wav_bytes(pcm.tobytes())


@pytest.fixture(autouse=True)
def _clean_injected_voice():
    """Every test starts with no fake STT/TTS installed."""
    set_transcriber(None)
    set_synthesizer(None)
    yield
    set_transcriber(None)
    set_synthesizer(None)


def read_until(ws, types, limit: int = 60) -> list[dict]:  # noqa: ANN001
    """Collect events until every type in ``types`` has been seen."""
    seen: list[dict] = []
    wanted = set(types)
    for _ in range(limit):
        msg = ws.receive_json()
        seen.append(msg)
        if wanted.issubset({m["type"] for m in seen}):
            break
    return seen


def types_of(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def read_until_listening(ws, limit: int = 60) -> list[dict]:  # noqa: ANN001
    """Collect events until the session returns to ``listening``.

    Every completed turn (and every pause-free recovery) ends with
    ``{"type": "phase", "state": "listening"}``, so this reads exactly
    one full turn's worth of events.
    """
    events: list[dict] = []
    for _ in range(limit):
        msg = ws.receive_json()
        events.append(msg)
        if msg == {"type": "phase", "state": "listening"}:
            break
    return events


# ---------------------------------------------------------------------------
# VAD unit tests
# ---------------------------------------------------------------------------


def test_vad_never_triggers_on_silence() -> None:
    vad = EnergyVAD()
    for _ in range(30):
        assert vad.process(silence_frame()) is False
    assert vad.speaking is False


def test_vad_detects_speech_after_silence() -> None:
    vad = EnergyVAD()
    for _ in range(5):
        vad.process(silence_frame())
    results = [vad.process(tone_frame()) for _ in range(5)]
    assert any(results) is True
    assert vad.speaking is True


def test_vad_hangover_keeps_one_turn_open() -> None:
    vad = EnergyVAD(hangover_frames=5)
    for _ in range(5):
        vad.process(silence_frame())
    for _ in range(5):
        vad.process(tone_frame())
    # A brief dip (between words) must not end the utterance...
    vad.process(silence_frame())
    assert vad.speaking is True
    # ...but sustained silence does.
    for _ in range(10):
        vad.process(silence_frame())
    assert vad.speaking is False


def test_vad_echo_margin_suppresses_playback_leakage() -> None:
    """While the AI speaks, only clearly louder audio counts as speech."""
    plain = EnergyVAD()
    echo = EnergyVAD()
    # The noise floor has learned the playback-leakage level (-60 dBFS):
    # with that floor, a borderline voice frame (-49 dBFS) sits above the
    # normal threshold (floor + 9 dB) but below the raised one (the echo
    # margin adds 10*log10(2.5) ~= 4 dB).
    plain._floor = -60.0
    echo._floor = -60.0
    borderline = tone_frame(amp=164)
    assert plain.process(borderline) is True
    echo.echo_active = True
    assert echo.process(borderline) is False
    # The same frame is still heard once the AI stops speaking.
    echo.echo_active = False
    echo.new_turn()
    assert echo.process(borderline) is True


def test_vad_new_turn_keeps_the_noise_floor() -> None:
    vad = EnergyVAD()
    for _ in range(5):
        vad.process(silence_frame())
    for _ in range(5):
        vad.process(tone_frame())
    floor = vad._floor
    vad.new_turn()
    assert vad.speaking is False
    assert vad._floor == floor  # the learned floor survives between turns
    vad.reset()
    assert vad._floor is None  # full reset (new connection) clears it


# ---------------------------------------------------------------------------
# WAV chunking (browser playback units)
# ---------------------------------------------------------------------------


def test_wav_chunks_are_playable_and_reassemble() -> None:
    pcm = tone_frame() * 50  # 1.5 s
    wav = wav_bytes(pcm)
    chunks = split_wav_chunks(wav, chunk_ms=400)
    assert len(chunks) >= 3
    total = 0
    for chunk in chunks:
        assert chunk[:4] == b"RIFF" and chunk[8:12] == b"WAVE"
        data_len = int.from_bytes(chunk[40:44], "little")
        assert data_len == len(chunk) - 44  # header size field is correct
        riff_len = int.from_bytes(chunk[4:8], "little")
        assert riff_len == len(chunk) - 8
        total += data_len
    assert total == len(pcm)


# ---------------------------------------------------------------------------
# WebSocket protocol (anonymous mode, faked LLM/STT/TTS)
# ---------------------------------------------------------------------------


def test_voice_start_returns_ready() -> None:
    with api_client(FakeLLM()) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"start"}')
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["tts"] in ("piper", "browser")
            ws.send_text('{"type":"ping"}')
            assert ws.receive_json() == {"type": "pong"}
            ws.send_text('{"type":"stop"}')
            assert ws.receive_json() == {"type": "stopped"}


def test_first_message_must_be_start() -> None:
    with api_client(FakeLLM()) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"ping"}')
            err = ws.receive_json()
            assert err["code"] == "bad_start"
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 4400


def test_full_voice_turn_runs_the_whole_pipeline() -> None:
    stt = ScriptedSTT(BENIGN)
    set_transcriber(stt)
    set_synthesizer(fake_tts)
    llm = FakeLLM()
    with api_client(llm) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"start"}')
            assert ws.receive_json()["type"] == "ready"
            ws.send_text(audio_msg(utterance()))
            events = read_until_listening(ws)
            names = types_of(events)
            phases = [e["state"] for e in events if e["type"] == "phase"]
            # The exact conversation order the UI depends on:
            # transcript -> thinking -> reply -> speaking -> listening.
            assert "user_transcript" in names
            assert phases[0] == "thinking"
            assert "reply" in names
            assert "tts_start" in names
            assert "tts_end" in names
            assert "speaking" in phases
            assert (
                names.index("user_transcript")
                < names.index("reply")
                < names.index("tts_start")
                < names.index("tts_end")
            )
            assert phases[-1] == "listening"
            # Transcription reached the real pipeline (faked LLM called
            # with exactly what the "user said").
            assert llm.calls and llm.calls[0]["message"] == BENIGN
            transcript = next(e for e in events if e["type"] == "user_transcript")
            assert transcript["text"] == BENIGN
            reply = next(e for e in events if e["type"] == "reply")
            assert reply["text"] == llm.text
            assert reply["response"]["source"] == "llm"
            audio = [e for e in events if e["type"] == "audio"]
            assert len(audio) >= 2
            # Back to listening for the next turn - hands-free loop.
            assert events[-1] == {"type": "phase", "state": "listening"}
            assert stt.calls == 1


def test_empty_transcript_never_reaches_the_model() -> None:
    stt = ScriptedSTT("")  # noise / silence recognised as nothing
    set_transcriber(stt)
    llm = FakeLLM()
    with api_client(llm) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"start"}')
            ws.receive_json()
            ws.send_text(audio_msg(utterance()))
            events = read_until(ws, ("skipped",), limit=10)
            assert events[-1] == {"type": "skipped", "reason": "no_speech"}
            assert "reply" not in types_of(events)
            assert llm.calls == []
            assert stt.calls == 1


def test_duplicate_transcript_gets_one_response() -> None:
    set_transcriber(ScriptedSTT(BENIGN, BENIGN))
    set_synthesizer(fake_tts)
    llm = FakeLLM()
    with api_client(llm) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"start"}')
            ws.receive_json()
            ws.send_text(audio_msg(utterance()))
            read_until(ws, ("reply", "tts_end"), limit=40)
            ws.send_text(audio_msg(utterance()))
            events = read_until(ws, ("skipped",), limit=20)
            assert events[-1]["type"] == "skipped"
            assert events[-1]["reason"] == "duplicate"
            assert len(llm.calls) == 1  # only one response to one speech


def test_interrupt_stops_audio_and_returns_to_listening() -> None:
    set_transcriber(ScriptedSTT(BENIGN))
    set_synthesizer(fake_tts)  # 2 s of audio = 5 chunks to interrupt
    with api_client(FakeLLM()) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"start"}')
            ws.receive_json()
            ws.send_text(audio_msg(utterance()))
            events = read_until(ws, ("audio",), limit=40)
            assert events[-1]["type"] == "audio"
            ws.send_text('{"type":"interrupt"}')
            tail = read_until(ws, ("tts_end", "phase"), limit=20)
            names = types_of(tail)
            assert "interrupted" in names
            end = next(e for e in tail if e["type"] == "tts_end")
            assert end["interrupted"] is True
            assert tail[-1] == {"type": "phase", "state": "listening"}


def test_barge_in_while_speaking_triggers_interrupt() -> None:
    """User speech during playback (past the echo margin) cancels TTS."""
    set_transcriber(ScriptedSTT(BENIGN, "and then I felt better"))
    set_synthesizer(fake_tts)
    llm = FakeLLM()
    with api_client(llm) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"start"}')
            ws.receive_json()
            ws.send_text(audio_msg(utterance()))
            read_until(ws, ("audio",), limit=40)  # AI starts speaking
            # Real voice over the playing audio: server-side barge-in.
            ws.send_text(audio_msg(utterance()))
            events = read_until(ws, ("interrupted",), limit=30)
            assert any(e["type"] == "interrupted" for e in events)
            # The barge-in utterance becomes the next turn (queued after
            # the interrupted reply), still hands-free.
            followup = read_until(ws, ("reply",), limit=60)
            replies = [e for e in followup if e["type"] == "reply"]
            assert replies and replies[-1]["text"] == llm.text
            assert len(llm.calls) == 2


def test_pause_and_resume_phases() -> None:
    set_transcriber(ScriptedSTT(BENIGN))
    with api_client(FakeLLM()) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"start"}')
            ws.receive_json()
            ws.send_text('{"type":"pause"}')
            assert ws.receive_json() == {"type": "phase", "state": "paused"}
            # Audio while paused is ignored entirely.
            ws.send_text(audio_msg(utterance()))
            ws.send_text('{"type":"ping"}')
            assert ws.receive_json() == {"type": "pong"}
            ws.send_text('{"type":"resume"}')
            assert ws.receive_json() == {"type": "phase", "state": "listening"}


def test_muted_microphone_sends_no_audio() -> None:
    set_transcriber(ScriptedSTT(BENIGN))
    llm = FakeLLM()
    with api_client(llm) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"start"}')
            ws.receive_json()
            ws.send_text('{"type":"mute","muted":true}')
            ws.send_text(audio_msg(utterance()))
            ws.send_text('{"type":"ping"}')
            assert ws.receive_json() == {"type": "pong"}  # nothing else arrived
            assert llm.calls == []
            ws.send_text('{"type":"mute","muted":false}')
            ws.send_text(audio_msg(utterance()))
            events = read_until(ws, ("reply",), limit=40)
            assert len(llm.calls) == 1


def test_stt_failure_is_reported_and_session_survives() -> None:
    def boom(audio: np.ndarray, language: str | None = None) -> str:
        raise STTError("model download failed")

    set_transcriber(boom)
    with api_client(FakeLLM()) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"start"}')
            ws.receive_json()
            ws.send_text(audio_msg(utterance()))
            events = read_until_listening(ws, limit=30)
            err = next(e for e in events if e["type"] == "error")
            assert err["code"] == "stt_failed"
            assert events[-1] == {"type": "phase", "state": "listening"}
            # The session survives an STT failure.
            ws.send_text('{"type":"ping"}')
            assert ws.receive_json() == {"type": "pong"}


def test_tts_failure_falls_back_to_browser() -> None:
    set_transcriber(ScriptedSTT(BENIGN))

    def no_voice(text: str) -> bytes:
        raise TTSError("no voice model")

    set_synthesizer(no_voice)
    with api_client(FakeLLM()) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"start"}')
            ws.receive_json()
            ws.send_text(audio_msg(utterance()))
            events = read_until_listening(ws, limit=30)
            names = types_of(events)
            assert "reply" in names  # the reply still arrived
            assert "tts_fallback" in names
            assert "audio" not in names  # no server audio was attempted
            assert events[-1] == {"type": "phase", "state": "listening"}


def test_malformed_and_unknown_messages_do_not_kill_the_session() -> None:
    with api_client(FakeLLM()) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"start"}')
            ws.receive_json()
            ws.send_text("{not json")
            assert ws.receive_json()["code"] == "bad_json"
            ws.send_text('{"type":"warp_drive"}')
            assert ws.receive_json()["code"] == "bad_type"
            ws.send_text('{"type":"ping"}')  # still alive
            assert ws.receive_json() == {"type": "pong"}


# ---------------------------------------------------------------------------
# Safety: voice is another transport, not another pipeline
# ---------------------------------------------------------------------------


def test_crisis_utterance_hits_the_same_pre_generation_gate() -> None:
    """A transcribed crisis message is blocked *before* the LLM, exactly
    like POST /api/chat - voice never bypasses the safety gate."""
    set_transcriber(ScriptedSTT(BLOCKER))
    set_synthesizer(fake_tts)
    llm = FakeLLM()
    with api_client(llm) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_text('{"type":"start"}')
            ws.receive_json()
            ws.send_text(audio_msg(utterance()))
            events = read_until(ws, ("reply",), limit=40)
            reply = next(e for e in events if e["type"] == "reply")
            assert reply["response"]["source"] == "pre_blocked"
            assert llm.calls == []  # the model was never consulted
            # The blocked reply is still spoken (it is the safe answer).
            assert reply["text"]


# ---------------------------------------------------------------------------
# Authentication + persistence (existing store, existing session)
# ---------------------------------------------------------------------------


def register(client: TestClient) -> dict:  # noqa: ANN001
    suffix = uuid4().hex[:10]
    response = client.post(
        "/api/auth/register",
        json={
            "display_name": "Voice Tester",
            "username": f"voice{suffix}",
            "email": f"voice{suffix}@example.com",
            "password": "voice-pass-123",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def messages(client: TestClient, token: str, session_id: str) -> list[dict]:  # noqa: ANN001
    response = client.get(
        f"/api/conversations/{session_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    # The endpoint returns an envelope {session_id, title, messages}; the
    # tests assert on the rows themselves.
    return response.json()["messages"]


def test_voice_requires_token_when_auth_enabled(tmp_path) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "voice.db"):
        with api_client(FakeLLM()) as client:
            with client.websocket_connect("/ws/voice") as ws:
                ws.send_text('{"type":"start"}')
                err = ws.receive_json()
                assert err["code"] == "unauthorized"
                with pytest.raises(WebSocketDisconnect) as exc:
                    ws.receive_json()
                assert exc.value.code == 4401


def test_voice_turns_persist_one_row_per_speaker_no_duplicates(tmp_path) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "voice.db"):
        with api_client(FakeLLM()) as client:
            token = register(client)["token"]
            set_transcriber(ScriptedSTT(BENIGN, "I felt calmer after breathing."))
            set_synthesizer(fake_tts)
            with client.websocket_connect("/ws/voice") as ws:
                ws.send_text(f'{{"type":"start","token":"{token}"}}')
                ready = ws.receive_json()
                assert ready["type"] == "ready"
                assert ready["session_id"] is None  # created by turn one

                ws.send_text(audio_msg(utterance()))
                first = read_until_listening(ws)
                session_id = next(
                    e for e in first if e["type"] == "reply"
                )["session_id"]
                assert session_id

                ws.send_text(audio_msg(utterance()))
                second = read_until_listening(ws)
                assert (
                    next(e for e in second if e["type"] == "reply")["session_id"]
                    == session_id
                )  # both turns on the SAME conversation
                ws.send_text('{"type":"stop"}')
                ws.receive_json()

            rows = messages(client, token, session_id)
            # Exactly one user row + one assistant row per turn: no
            # duplicates from the voice transport.
            assert len(rows) == 4
            assert [r["role"] for r in rows] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]
            assert rows[0]["content"] == BENIGN
            assert rows[2]["content"] == "I felt calmer after breathing."


def test_voice_reuses_an_existing_session(tmp_path) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "voice2.db"):
        with api_client(FakeLLM()) as client:
            token = register(client)["token"]
            # Start a conversation over HTTP first (existing behaviour).
            created = client.post(
                "/api/chat",
                json={"message": BENIGN},
                headers={"Authorization": f"Bearer {token}"},
            ).json()
            session_id = created["session_id"]
            set_transcriber(ScriptedSTT("thanks, that helps"))
            set_synthesizer(fake_tts)
            with client.websocket_connect("/ws/voice") as ws:
                ws.send_text(
                    f'{{"type":"start","token":"{token}",'
                    f'"session_id":"{session_id}"}}'
                )
                ready = ws.receive_json()
                assert ready["session_id"] == session_id
                ws.send_text(audio_msg(utterance()))
                events = read_until(ws, ("reply",), limit=40)
                reply = next(e for e in events if e["type"] == "reply")
                assert reply["session_id"] == session_id
                ws.send_text('{"type":"stop"}')
            rows = messages(client, token, session_id)
            # 1 HTTP turn (2 rows) + 1 voice turn (2 rows) = 4, in order.
            assert [r["role"] for r in rows] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]


def test_foreign_session_id_is_rejected_like_http(tmp_path) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "voice3.db"):
        with api_client(FakeLLM()) as client:
            owner = register(client)["token"]
            other = register(client)["token"]
            session_id = client.post(
                "/api/chat",
                json={"message": BENIGN},
                headers={"Authorization": f"Bearer {owner}"},
            ).json()["session_id"]
            with client.websocket_connect("/ws/voice") as ws:
                ws.send_text(
                    f'{{"type":"start","token":"{other}",'
                    f'"session_id":"{session_id}"}}'
                )
                err = ws.receive_json()
                assert err["code"] == "conversation_not_found"
                with pytest.raises(WebSocketDisconnect) as exc:
                    ws.receive_json()
                assert exc.value.code == 4400


def test_anonymous_mode_works_without_token() -> None:
    """AUTH_ENABLED=false: voice runs without a token, one shared session."""
    from app.config.settings import reset_settings_cache

    old = os.environ.get("AUTH_ENABLED")
    os.environ["AUTH_ENABLED"] = "false"
    reset_settings_cache()
    try:
        set_transcriber(ScriptedSTT(BENIGN, "thank you for listening"))
        set_synthesizer(fake_tts)
        llm = FakeLLM()
        with api_client(llm) as client:
            with client.websocket_connect("/ws/voice") as ws:
                ws.send_text('{"type":"start"}')
                assert ws.receive_json()["type"] == "ready"
                ws.send_text(audio_msg(utterance()))
                events = read_until(ws, ("reply",), limit=40)
                first = next(e for e in events if e["type"] == "reply")
                # Two turns stay on one anonymous session.
                ws.send_text(audio_msg(utterance()))
                events2 = read_until(ws, ("reply",), limit=40)
                second = next(e for e in events2 if e["type"] == "reply")
                assert first["session_id"] == second["session_id"]
                assert len(llm.calls) == 2
    finally:
        if old is None:
            os.environ.pop("AUTH_ENABLED", None)
        else:
            os.environ["AUTH_ENABLED"] = old
        reset_settings_cache()
