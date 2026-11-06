"""Text-to-speech via Piper (Phase 24). [OURS]

Neural, offline, free synthesis of the assistant reply:

* ``piper-tts`` with a downloaded voice model (default
  ``en_US-lessac-medium``) under ``VOICE_TTS_CACHE_DIR`` (default
  ``data/voice/tts``) - ``ensure_voice()`` downloads it once;
* output is 16-bit mono WAV bytes, streamed to the browser in chunks so
  the client can start playback before synthesis finishes and can stop
  instantly on barge-in;
* when Piper or the voice model is unavailable, ``synthesize`` raises
  ``TTSError`` and the WebSocket tells the client to fall back to the
  browser's built-in SpeechSynthesis (``{"tts": "browser"}``) - the
  conversation never breaks because a model is missing.

``set_synthesizer`` exists for tests to inject a fake.
"""

from __future__ import annotations

import io
import struct
import threading
import wave
from collections.abc import Callable
from typing import Any

from app.config.logging_config import get_logger
from app.config.settings import get_settings

logger = get_logger(__name__)


class TTSError(RuntimeError):
    """Synthesis failed; ``message`` is safe to show the user."""


# A synthesizer takes text and returns WAV bytes (16 kHz mono int16).
Synthesizer = Callable[[str], bytes]

_synthesizer: Synthesizer | None = None
_lock = threading.Lock()


def set_synthesizer(fn: Synthesizer | None) -> None:
    """Inject (or clear) a synthesizer - used by the test suite."""
    global _synthesizer
    with _lock:
        _synthesizer = fn


class _PiperSynthesizer:
    """Lazy Piper singleton with a one-time voice download."""

    def __init__(self) -> None:
        self._voice: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        settings = get_settings()
        cache_dir = settings.voice_tts_cache_path
        cache_dir.mkdir(parents=True, exist_ok=True)
        voice_path = cache_dir / f"{settings.voice_tts_voice}.onnx"
        try:
            from piper import PiperVoice  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - env dependent
            raise TTSError(
                "Speech synthesis is not installed on the server "
                "(pip install piper-tts)."
            ) from exc
        if not voice_path.exists() or not voice_path.with_suffix(".onnx.json").exists():
            logger.info("Downloading Piper voice %s", settings.voice_tts_voice)
            try:
                from piper.download_voices import download_voice  # type: ignore

                download_voice(settings.voice_tts_voice, cache_dir)
            except Exception as exc:  # noqa: BLE001 - network/IO failures
                raise TTSError(
                    f"Voice model {settings.voice_tts_voice} is not "
                    f"available for download: {exc}"
                ) from exc
        try:
            return PiperVoice.load(str(voice_path))
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Could not load Piper voice: {exc}") from exc

    def __call__(self, text: str) -> bytes:
        with self._lock:
            if self._voice is None:
                self._voice = self._load()
            voice = self._voice
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                voice.synthesize_wav(text, w)
            return buf.getvalue()
        except TTSError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Speech synthesis failed: {exc}") from exc


_default = _PiperSynthesizer()


def wav_bytes(pcm: bytes, *, sample_rate: int = 16_000) -> bytes:
    """Wrap raw int16 mono PCM in a minimal WAV container."""
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return out.getvalue()


def synthesize(text: str) -> bytes:
    """Reply text -> WAV bytes. Raises ``TTSError`` when unavailable."""
    text = (text or "").strip()
    if not text:
        raise TTSError("Nothing to speak.")
    with _lock:
        synthesizer = _synthesizer
    if synthesizer is None:
        synthesizer = _default
    data = synthesizer(text)
    if not data:
        raise TTSError("Synthesis produced no audio.")
    return data


def tts_available() -> bool:
    """Best-effort probe: is a backend synthesizer importable?"""
    with _lock:
        if _synthesizer is not None:
            return True
    try:
        import piper  # noqa: F401

        return True
    except ImportError:
        return False


def split_wav_chunks(
    wav: bytes, chunk_ms: int = 400, sample_rate: int = 16_000
) -> list[bytes]:
    """Split a WAV byte blob into ~``chunk_ms`` mini-WAV chunks.

    Each chunk is a complete standard 44-byte-header WAV with the same
    format (mono, 16-bit, ``sample_rate``) so the browser can decode
    each one independently with ``AudioContext.decodeAudioData`` and
    play them back-to-back.
    """
    if len(wav) < 44:
        return [wav]
    # Find the data chunk (skip RIFF header + any extra chunks).
    pos, data_start = 12, None
    while pos + 8 <= len(wav):
        cid = wav[pos : pos + 4]
        clen = int.from_bytes(wav[pos + 4 : pos + 8], "little")
        if cid == b"data":
            data_start = pos + 8
            break
        pos += 8 + clen + (clen & 1)
    if data_start is None:
        return [wav]
    pcm = wav[data_start:]
    # Bytes per chunk = rate * 2 (mono int16) * ms / 1000, aligned to 2.
    per = max(int(sample_rate * 2 * chunk_ms / 1000) & ~1, 2)
    chunks: list[bytes] = []
    for off in range(0, len(pcm), per):
        piece = pcm[off : off + per]
        header = _wav_header(len(piece), sample_rate)
        chunks.append(header + piece)
    return chunks or [wav]


def _wav_header(data_len: int, sample_rate: int) -> bytes:
    """Standard 44-byte RIFF/WAVE header for mono 16-bit PCM."""
    byte_rate = sample_rate * 2
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_len,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        byte_rate,
        2,
        16,
        b"data",
        data_len,
    )
