"""Voice conversation WebSocket (Phase 24). [OURS]

``/ws/voice`` turns the existing chat pipeline into a hands-free,
two-way voice conversation:

    mic PCM -> VAD turn detection -> Whisper STT -> **perform_turn()**
            -> Piper TTS -> streamed WAV chunks -> speaker -> re-listen

Design rules (all deliberate):

* **Same pipeline, different transport.** Every transcribed utterance is
  executed through ``app.api.chat.perform_turn`` - the exact function
  ``POST /api/chat`` runs - so auth, session ownership, persistence,
  sentiment/emotion/risk analysis, personalization, memory and the
  pre/post generation safety gates apply unchanged. Voice never bypasses
  anything.
* **Auth on the first message.** Browsers cannot set an Authorization
  header on a WebSocket, so the client sends ``{"type":"start",
  "token":...}``; an invalid token closes the socket with 4401.
* **One reader task, one worker task.** The receive loop owns VAD and
  control messages (interrupt/pause/mute/stop); a worker consumes
  finalized utterances in order - no duplicate or concurrent replies.
* **Barge-in cancels playback.** ``interrupt`` (from the client's local
  echo check) or server-side VAD while ``speaking`` sets an event the
  worker checks between audio chunks - the stream stops immediately and
  listening resumes. A reply that was already persisted is never
  re-synthesised.
* **Noise never reaches the model.** Empty/whisper-noise transcripts
  are dropped before the pipeline (no LLM call, no DB rows), and a
  repeated transcript within a few seconds is deduplicated.
* **Graceful degradation.** Missing Whisper/Piper models surface as
  ``error``/``tts_fallback`` events; the browser can fall back to
  Web Speech + SpeechSynthesis without dropping the conversation.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.api.chat import perform_turn
from app.auth.store import get_auth_store
from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.models.schemas import SERVER_MAX_MESSAGE_CHARS, ChatRequest
from app.voice.stt import STTError, transcribe_pcm
from app.voice.tts import TTSError, split_wav_chunks, synthesize, tts_available
from app.voice.vad import EnergyVAD

logger = get_logger(__name__)
router = APIRouter()

# Whisper language codes we accept from the UI (Phase 5 locales).
_LANGUAGES = {"en", "te", "hi", "ta", "kn"}
# Close codes (application range 4000-4999).
_CLOSE_UNAUTHORIZED = 4401
_CLOSE_PROTOCOL = 4400
_CLOSE_INTERNAL = 4500

_BYTES_PER_SECOND = 16_000 * 2  # 16 kHz mono int16


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class VoiceSession:
    """One connected voice conversation (state machine + worker)."""

    def __init__(self, ws: WebSocket, user: dict | None, language: str) -> None:
        self.ws = ws
        self.user = user
        self.language = language
        self.settings = get_settings()
        self.session_id: str | None = None  # resolved from the first reply
        self.state = "listening"  # listening | thinking | speaking | paused
        self.muted = False
        self.stopped = False
        # Barge-in / pause / stop signal for the worker.
        self.cancel_speaking = asyncio.Event()
        # Finalized utterances (PCM) and typed messages, processed in order.
        self.turn_queue: asyncio.Queue[tuple[str, str | bytes]] = asyncio.Queue()
        # VAD (per-connection adaptive noise floor).
        self.vad = EnergyVAD(
            threshold_db=self.settings.voice_vad_threshold_db,
            hangover_frames=self.settings.voice_vad_hangover,
            frame_ms=self.settings.voice_frame_ms,
        )
        self._utterance = bytearray()
        self._speech_started = False
        self._last_text = ""
        self._last_text_at = 0.0
        self._worker: asyncio.Task | None = None
        self._recv_stopped = asyncio.Event()

    # ------------------------------------------------------------------
    async def send(self, payload: dict) -> None:
        try:
            await self.ws.send_text(json.dumps(payload))
        except (WebSocketDisconnect, RuntimeError):  # client went away
            self.stopped = True

    async def send_phase(self, state: str) -> None:
        if self.state != state:
            self.state = state
            await self.send({"type": "phase", "state": state})

    async def force_listening(self) -> None:
        """End a turn with an explicit ``phase listening`` event.

        Used after terminal outcomes (empty transcript, dedupe, errors)
        where the state may already *be* listening - the UI indicator
        still gets one deterministic reset signal per turn.
        """
        if self.stopped or self.state == "paused":
            return
        self.state = "listening"
        await self.send({"type": "phase", "state": "listening"})

    # ------------------------------------------------------------------
    # Receiving: VAD + control messages (the only reader of the socket).
    async def receive_loop(self) -> None:
        try:
            while not self.stopped:
                raw = await self.ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self.send({"type": "error", "code": "bad_json", "detail": "Malformed message."})
                    continue
                kind = msg.get("type")
                if kind == "audio":
                    await self._on_audio(msg)
                elif kind == "interrupt":
                    await self._on_interrupt()
                elif kind == "mute":
                    self.muted = bool(msg.get("muted", True))
                    if not self.muted and self.state == "paused":
                        await self.send_phase("listening")
                elif kind == "pause":
                    await self._on_pause()
                elif kind == "resume":
                    if self.state == "paused":
                        self.vad.new_turn()
                        self._utterance.clear()
                        self._speech_started = False
                        await self.send_phase("listening")
                elif kind == "text":
                    text = str(msg.get("message", "")).strip()
                    if text:
                        await self.turn_queue.put(("text", text[:SERVER_MAX_MESSAGE_CHARS]))
                elif kind == "stop":
                    self.stopped = True
                    await self.send({"type": "stopped"})
                elif kind == "ping":
                    await self.send({"type": "pong"})
                elif kind != "start":
                    await self.send({"type": "error", "code": "bad_type", "detail": f"Unknown type {kind!r}."})
        except WebSocketDisconnect:
            pass
        finally:
            self.stopped = True
            self.cancel_speaking.set()
            await self.turn_queue.put(("stop", b""))
            self._recv_stopped.set()

    async def _on_interrupt(self) -> None:
        """Client (or server VAD) says the user is talking over the AI."""
        self.cancel_speaking.set()
        if self.state in ("speaking", "thinking"):
            await self.send({"type": "interrupted"})

    async def _on_pause(self) -> None:
        self.cancel_speaking.set()
        self.vad.new_turn()
        self._utterance.clear()
        self._speech_started = False
        if self.state != "paused":
            await self.send_phase("paused")

    async def _on_audio(self, msg: dict) -> None:
        if self.stopped or self.muted or self.state == "paused":
            return
        try:
            pcm = base64.b64decode(msg.get("data", ""), validate=True)
        except Exception:  # noqa: BLE001 - corrupt frame, drop it
            return
        if not pcm:
            return
        # Frame by frame: a whole utterance can arrive in one message, so
        # accumulate/finalize decisions must happen per frame, not per
        # batch (a batch would see only the end state of the hangover).
        frame_len = self.vad.frame_samples * 2
        for off in range(0, len(pcm), frame_len):
            frame = pcm[off : off + frame_len]
            if len(frame) < 2:
                break
            raw = self.vad.process(frame)

            # Barge-in: user talks while the AI's audio is playing. The
            # echo margin already raised the threshold, so this is a real
            # voice, not playback leakage.
            if self.state == "speaking" and raw and not self.muted:
                self.cancel_speaking.set()
                self.vad.echo_active = False
                await self.send({"type": "interrupted"})
                await self.send_phase("listening")

            if self.state in ("speaking", "paused") and not self._speech_started:
                continue  # discard playback leakage frames

            if raw:
                self._speech_started = True
            if self._speech_started and self.vad.speaking:
                self._utterance.extend(frame)
                cap = int(self.settings.voice_max_utterance_s * _BYTES_PER_SECOND)
                if len(self._utterance) >= cap:
                    await self._finalize_utterance()
            elif self._speech_started and not self.vad.speaking:
                # Hangover expired: the turn is over.
                await self._finalize_utterance()

    async def _finalize_utterance(self) -> None:
        pcm = bytes(self._utterance)
        self._utterance = bytearray()
        self._speech_started = False
        if pcm:
            await self.turn_queue.put(("utt", pcm))

    # ------------------------------------------------------------------
    # Worker: STT -> pipeline -> TTS, in queue order (never concurrent).
    async def worker_loop(self) -> None:
        pad = b"\x00\x00" * int(
            self.settings.voice_padding_ms * 16_000 / 1000
        )
        while True:
            kind, payload = await self.turn_queue.get()
            if kind == "stop" or self.stopped:
                return
            try:
                if kind == "utt":
                    await self._handle_utterance(pad + payload + pad)
                elif kind == "text":
                    await self._handle_text(str(payload), transcript=None)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad turn must not kill the session
                logger.exception("Voice turn failed")
                await self.send(
                    {"type": "error", "code": "chat_failed", "detail": "The turn could not be completed."}
                )
                await self.force_listening()

    async def _handle_utterance(self, pcm: bytes) -> None:
        try:
            text = await asyncio.to_thread(
                transcribe_pcm, pcm, language=self.language
            )
        except STTError as exc:
            await self.send(
                {"type": "error", "code": "stt_failed", "detail": str(exc)}
            )
            await self.force_listening()
            return
        if not text:
            # Silence / noise / unintelligible: never call the model.
            await self.send({"type": "skipped", "reason": "no_speech"})
            await self.force_listening()
            return
        await self._handle_text(text, transcript=text)

    async def _handle_text(self, text: str, transcript: str | None) -> None:
        # Dedupe: the same words recognised twice in quick succession
        # (echo, double VAD trigger) must produce one response only.
        now = time.monotonic()
        if text == self._last_text and now - self._last_text_at < 6.0:
            await self.send({"type": "skipped", "reason": "duplicate"})
            await self.force_listening()
            return
        self._last_text, self._last_text_at = text, now

        if transcript is not None:
            await self.send({"type": "user_transcript", "text": transcript})
        self.cancel_speaking.clear()
        await self.send_phase("thinking")

        try:
            response = await asyncio.to_thread(
                perform_turn,
                ChatRequest(message=text, session_id=self.session_id),
                self.user,
            )
        except ValidationError:
            await self.send(
                {"type": "error", "code": "bad_message", "detail": "Message empty or too long."}
            )
            await self.force_listening()
            return
        except Exception as exc:  # noqa: BLE001 - includes HTTPException(404)
            if isinstance(exc, HTTPException):
                await self.send(
                    {"type": "error", "code": "conversation_not_found", "detail": str(exc.detail)}
                )
            else:
                logger.exception("Voice chat turn failed")
                await self.send(
                    {"type": "error", "code": "chat_failed", "detail": "The turn could not be completed."}
                )
            await self.force_listening()
            return

        self.session_id = response.session_id
        await self.send(
            {
                "type": "reply",
                "session_id": response.session_id,
                "text": response.reply,
                "response": response.model_dump(mode="json"),
            }
        )
        if self.stopped or self.state == "paused":
            return  # no audio while paused; the transcript is on screen
        if self.cancel_speaking.is_set():
            # User barged in while we were thinking: skip the audio, the
            # transcript is already on screen and persisted.
            await self.force_listening()
            return
        await self._speak(response.reply)

    async def _speak(self, text: str) -> None:
        if self.stopped or self.state == "paused" or self.cancel_speaking.is_set():
            await self.force_listening()
            return
        await self.send_phase("speaking")
        self.vad.echo_active = True
        try:
            try:
                wav = await asyncio.to_thread(synthesize, text)
            except TTSError as exc:
                # No Piper voice: the browser reads the reply aloud.
                await self.send({"type": "tts_fallback", "detail": str(exc)})
                await self._after_speech()
                return
            await self.send({"type": "tts_start"})
            chunks = split_wav_chunks(
                wav, chunk_ms=self.settings.voice_tts_chunk_ms
            )
            # Stream at playback pace: the interrupt window must last as
            # long as the audio itself (and the client never underruns).
            pace = self.settings.voice_tts_chunk_ms / 1000.0
            interrupted = False
            for i, chunk in enumerate(chunks):
                if self.cancel_speaking.is_set() or self.stopped:
                    interrupted = True
                    break
                await self.send(
                    {"type": "audio", "seq": i, "data": _b64(chunk)}
                )
                if i < len(chunks) - 1:
                    await asyncio.sleep(pace)
            if not interrupted and (self.cancel_speaking.is_set() or self.stopped):
                interrupted = True
            await self.send({"type": "tts_end", "interrupted": interrupted})
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - never kill the session on TTS
            logger.exception("Voice TTS streaming failed")
            await self.send({"type": "tts_fallback", "detail": "Speech synthesis failed."})
        finally:
            await self._after_speech()

    async def _after_speech(self) -> None:
        self.vad.echo_active = False
        if not self._speech_started:
            # No new utterance in progress: clear the turn state. When a
            # barge-in utterance is already accumulating we must NOT touch
            # it - wiping it here would truncate what the user is saying.
            self.vad.new_turn()
            self._utterance.clear()
        if not self.stopped and self.state != "paused":
            await self.send_phase("listening")


# ---------------------------------------------------------------------------
@router.websocket("/ws/voice")
async def voice_socket(ws: WebSocket) -> None:
    """Hands-free voice conversation (Phase 24)."""
    settings = get_settings()
    await ws.accept()
    if not settings.voice_enabled:
        await ws.send_text(
            json.dumps({"type": "error", "code": "voice_disabled", "detail": "Voice mode is disabled."})
        )
        await ws.close(code=_CLOSE_PROTOCOL)
        return

    # First frame must be `start` (auth + language + optional session).
    try:
        first = json.loads(await ws.receive_text())
    except (json.JSONDecodeError, WebSocketDisconnect):
        await ws.close(code=_CLOSE_PROTOCOL)
        return
    if first.get("type") != "start":
        await ws.send_text(
            json.dumps({"type": "error", "code": "bad_start", "detail": "First message must be start."})
        )
        await ws.close(code=_CLOSE_PROTOCOL)
        return

    user: dict | None = None
    if settings.auth_enabled:
        token = first.get("token") or ""
        user = get_auth_store().user_for_token(token) if token else None
        if user is None:
            await ws.send_text(
                json.dumps({"type": "error", "code": "unauthorized", "detail": "Invalid or missing token."})
            )
            await ws.close(code=_CLOSE_UNAUTHORIZED)
            return

    language = str(first.get("language") or settings.voice_stt_language)
    if language not in _LANGUAGES:
        language = settings.voice_stt_language

    session = VoiceSession(ws, user, language)

    # A client-supplied session must exist and belong to the caller
    # (same rule as POST /api/chat); otherwise the first turn starts a
    # fresh conversation and its id comes back on the first `reply`.
    requested = first.get("session_id")
    if isinstance(requested, str) and requested:
        if user is None:
            session.session_id = requested  # anonymous mode: no ownership
        else:
            owner = get_auth_store().conversation_owner(requested)
            if owner is None or owner != user["id"]:
                # Mirrors POST /api/chat: unknown/foreign ids are a 404.
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "code": "conversation_not_found",
                            "detail": requested,
                        }
                    )
                )
                await ws.close(code=_CLOSE_PROTOCOL)
                return
            session.session_id = requested
    elif user is None:
        # Anonymous mode: keep one state-engine session for the whole
        # voice conversation (there is no DB row to carry it).
        session.session_id = uuid4().hex

    await session.send(
        {
            "type": "ready",
            "session_id": session.session_id,
            "language": language,
            "tts": "piper" if tts_available() else "browser",
            "stt_model": settings.voice_stt_model,
            "frame_ms": settings.voice_frame_ms,
        }
    )

    session._worker = asyncio.create_task(session.worker_loop())
    try:
        await session.receive_loop()
    finally:
        session.stopped = True
        session.cancel_speaking.set()
        if session._worker is not None:
            session._worker.cancel()
            try:
                await session._worker
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        try:
            await ws.close(code=1000)
        except (RuntimeError, WebSocketDisconnect):
            pass
