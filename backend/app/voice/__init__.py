"""Voice conversation layer (Phase 24). [OURS]

An *additional* input/output interface on top of the existing chat
pipeline - it never replaces or bypasses it:

* ``vad.py`` - adaptive-energy voice activity detection over 16 kHz PCM
  frames (turn detection for incoming audio, echo margin while the AI
  speaks);
* ``stt.py``  - faster-whisper speech-to-text (lazy, offline, free);
* ``tts.py``  - Piper neural text-to-speech with a graceful fallback
  flag when no voice model is installed.

All transcription output is fed into ``app.api.chat.perform_turn`` - the
same auth, persistence, state, sentiment/emotion/risk analysis and
pre/post safety gates as ``POST /api/chat``.
"""
