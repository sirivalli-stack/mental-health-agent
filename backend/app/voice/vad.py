"""Adaptive-energy voice activity detection. [OURS] (Phase 24)

Turn detection for the voice WebSocket: no model weights, no network -
a per-connection adaptive noise floor over 16 kHz mono PCM frames.

Design:

* **Adaptive floor.** The noise floor tracks the running minimum of
  recent frame energies with a slow update, so background hum is learned
  and a sustained voice raises the speech threshold above it.
* **Hangover frames.** Speech that dips briefly (between words) stays
  ``speaking`` for ``hangover_frames`` frames - one utterance becomes
  one turn, not twenty word fragments.
* **Echo margin.** While the AI is talking (TTS audio playing on the
  same device) the threshold is raised by ``echo_factor`` so the mic
  does not immediately hear the speaker and barge in on itself; real
  barge-in still works because a user speaking over the reply is far
  louder than the played-back audio leakage.

Pure numpy on plain ``bytes``/``array`` frames - unit-testable with
synthetic signals.
"""

from __future__ import annotations

import numpy as np


class EnergyVAD:
    """Frame-by-frame voice activity detector.

    Parameters
    ----------
    threshold_db:
        Minimum frame energy (dB above the adaptive floor) to count as
        speech.
    hangover_frames:
        Frames of silence tolerated before a turn is declared ended.
    echo_factor:
        Multiplier applied to the threshold while ``echo_active`` is
        True (AI speaking) - raises the bar so playback leakage does not
        register as the user talking.
    floor_decay:
        How fast the noise floor follows *down* (per frame, 0-1). Slow
        on purpose: a loud burst must not instantly become "floor".
    """

    def __init__(
        self,
        *,
        threshold_db: float = 9.0,
        hangover_frames: int = 12,
        echo_factor: float = 2.5,
        floor_decay: float = 0.002,
        sample_rate: int = 16_000,
        frame_ms: int = 30,
    ) -> None:
        self.threshold_db = threshold_db
        self.hangover_frames = hangover_frames
        self.echo_factor = echo_factor
        self.floor_decay = floor_decay
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self._floor: float | None = None
        self._hangover = 0
        self.echo_active = False
        self.speaking = False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._floor = None
        self._hangover = 0
        self.echo_active = False
        self.speaking = False

    def new_turn(self) -> None:
        """End any utterance state but keep the learned noise floor."""
        self._hangover = 0
        self.speaking = False

    def _to_mono_float(self, frame: bytes | np.ndarray) -> np.ndarray:
        """Int16 PCM bytes (or an array) -> float32 mono in [-1, 1]."""
        if isinstance(frame, (bytes, bytearray, memoryview)):
            audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        else:
            audio = np.asarray(frame, dtype=np.float32)
        return audio / 32768.0 if audio.size and audio.max() > 1.5 else audio

    def frame_energy_db(self, frame: bytes | np.ndarray) -> float:
        """RMS energy of one frame in dBFS (silence -> -120 floor)."""
        audio = self._to_mono_float(frame)
        if audio.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return 20.0 * np.log10(max(rms, 1e-6))

    # ------------------------------------------------------------------
    def process(self, frame: bytes | np.ndarray) -> bool:
        """Feed one frame; return True while (or newly) speech is active."""
        energy = self.frame_energy_db(frame)

        # Adaptive noise floor: tracks quiet frames quickly (a cough must
        # not become the new floor either) and rises slowly, so the
        # speech threshold stays *relative* to the environment.
        if self._floor is None:
            self._floor = energy
        elif energy < self._floor:
            self._floor += (energy - self._floor) * max(self.floor_decay * 5.0, 0.02)
        else:
            self._floor += (energy - self._floor) * self.floor_decay

        threshold = self._floor + self.threshold_db
        if self.echo_active:
            threshold += 10.0 * np.log10(max(self.echo_factor, 1.0))

        if energy >= threshold and energy > self._floor + 3.0:
            self.speaking = True
            self._hangover = self.hangover_frames
        elif self.speaking:
            self._hangover -= 1
            if self._hangover <= 0:
                self.speaking = False
        return self.speaking

    def process_bytes(self, pcm: bytes) -> list[bool]:
        """Feed a concatenated PCM chunk, split into frames."""
        frame_len = self.frame_samples * 2  # int16 = 2 bytes/sample
        return [
            self.process(pcm[i : i + frame_len])
            for i in range(0, len(pcm), frame_len)
            if len(pcm[i : i + frame_len]) >= frame_len // 2
        ]


def speech_ratio(frames: list[bool]) -> float:
    """Fraction of frames classified as speech (0.0 when empty)."""
    if not frames:
        return 0.0
    return sum(1 for f in frames if f) / len(frames)
