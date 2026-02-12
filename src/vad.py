"""
Voice activity detection: simple energy-based end-of-utterance.
Optional Silero VAD can be added later.
"""

from __future__ import annotations

import numpy as np


def rms(audio: np.ndarray) -> float:
    """Root-mean-square of float32 audio."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64)))))


class EnergyVAD:
    """
    End-of-utterance when silence (RMS below threshold) lasts at least
    silence_duration seconds. Uses a fixed frame length for smoothing.
    """

    def __init__(
        self,
        sample_rate: int,
        silence_duration: float = 0.8,
        energy_threshold: float = 0.01,
        frame_seconds: float = 0.05,
    ):
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.energy_threshold = energy_threshold
        self.frame_samples = max(1, int(sample_rate * frame_seconds))
        self._silence_frames = 0
        self._speech_seen = False
        self._silence_frames_needed = max(
            1, int(silence_duration * sample_rate / self.frame_samples)
        )

    def process(self, audio: np.ndarray) -> bool:
        """
        Process a chunk. Returns True if end-of-utterance was detected
        (speech followed by enough silence).
        """
        audio = np.asarray(audio, dtype=np.float32).ravel()
        n = len(audio)
        for start in range(0, n, self.frame_samples):
            frame = audio[start : start + self.frame_samples]
            if len(frame) < self.frame_samples:
                break
            e = rms(frame)
            if e >= self.energy_threshold:
                self._speech_seen = True
                self._silence_frames = 0
            else:
                if self._speech_seen:
                    self._silence_frames += 1
                    if self._silence_frames >= self._silence_frames_needed:
                        self._speech_seen = False
                        self._silence_frames = 0
                        return True
        return False

    def reset(self) -> None:
        self._silence_frames = 0
        self._speech_seen = False
