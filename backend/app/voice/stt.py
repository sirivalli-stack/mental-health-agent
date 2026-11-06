"""Speech-to-text via faster-whisper (Phase 24). [OURS]

Offline, free, CPU-friendly transcription of the audio the browser sends
over the voice WebSocket:

* ``faster-whisper`` (CTranslate2) with the ``base`` model by default -
  ~74 MB downloaded once from Hugging Face into ``VOICE_STT_CACHE_DIR``
  (default ``data/voice/stt``) and then fully offline;
* int8 quantisation on CPU - fast enough for real-time turn-taking;
* lazy, process-wide singleton: the model loads once on the first
  transcription and stays warm;
* every failure (missing dependency, failed download, bad audio) raises
  ``STTError`` with a human message - the WebSocket reports it to the
  client instead of crashing the session.

``transcribe_pcm`` accepts raw 16 kHz mono int16 PCM (what the browser
captures). ``set_transcriber`` exists for tests to inject a fake.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from app.config.logging_config import get_logger
from app.config.settings import get_settings

logger = get_logger(__name__)


class STTError(RuntimeError):
    """Transcription failed; ``message`` is safe to show the user."""


# A transcriber takes 16 kHz float32 mono audio (and an optional
# language code) and returns text.
Transcriber = Callable[[np.ndarray, str | None], str]

_transcriber: Transcriber | None = None
_lock = threading.Lock()


def set_transcriber(fn: Transcriber | None) -> None:
    """Inject (or clear) a transcriber - used by the test suite."""
    global _transcriber
    with _lock:
        _transcriber = fn


class _WhisperTranscriber:
    """Lazy faster-whisper singleton (loaded once per process)."""

    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        settings = get_settings()
        cache_dir = str(settings.voice_stt_cache_path)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - env dependent
            raise STTError(
                "Speech recognition is not installed on the server "
                "(pip install faster-whisper)."
            ) from exc
        try:
            return WhisperModel(
                settings.voice_stt_model,
                device="cpu",
                compute_type="int8",
                download_root=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 - download/IO failures
            raise STTError(
                f"Could not load the speech model "
                f"({settings.voice_stt_model}): {exc}"
            ) from exc

    def __call__(self, audio: np.ndarray, language: str | None = None) -> str:
        with self._lock:
            if self._model is None:
                logger.info("Loading Whisper model %s", get_settings().voice_stt_model)
                self._model = self._load()
            model = self._model
        settings = get_settings()
        lang = (language or settings.voice_stt_language or "en") or None
        try:
            segments, _info = model.transcribe(
                audio,
                language=lang,
                beam_size=1,
                vad_filter=False,  # our own VAD already segmented the audio
                condition_on_previous_text=False,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        except STTError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as a clean error
            raise STTError(f"Speech recognition failed: {exc}") from exc


_default = _WhisperTranscriber()


def transcribe_pcm(
    pcm: bytes, *, sample_rate: int = 16_000, language: str | None = None
) -> str:
    """Raw int16 mono PCM -> text ("" when nothing intelligible)."""
    if not pcm or len(pcm) < 2:
        return ""
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return transcribe_audio(audio, sample_rate=sample_rate, language=language)


def transcribe_audio(
    audio: np.ndarray, *, sample_rate: int = 16_000, language: str | None = None
) -> str:
    """Float32 mono audio -> text. Raises ``STTError`` on failure."""
    if audio.size == 0:
        return ""
    with _lock:
        transcriber = _transcriber
    if transcriber is None:
        transcriber = _default
    text = (transcriber(audio, language) or "").strip()
    # Whisper sometimes returns only punctuation/whitespace for noise.
    if not any(ch.isalnum() for ch in text):
        return ""
    return text
