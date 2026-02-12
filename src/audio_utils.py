"""
Audio utilities: resampling, stereo→mono, and a ring buffer for VAD/ASR.
"""

from __future__ import annotations

import numpy as np


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D float32 audio from orig_sr to target_sr (linear interpolation)."""
    if orig_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    duration = len(audio) / orig_sr
    n_target = int(round(duration * target_sr))
    if n_target == 0:
        return np.array([], dtype=np.float32)
    indices = np.linspace(0, len(audio) - 1, n_target, endpoint=True)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def stereo_to_mono(audio: np.ndarray, channels: int = 2) -> np.ndarray:
    """Convert interleaved stereo to mono (average)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1 and channels >= 2:
        n = len(audio) // channels
        audio = audio[: n * channels].reshape(n, channels)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def prepare_for_asr(
    audio: np.ndarray,
    orig_sr: int,
    orig_channels: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Convert raw mic chunk to 16 kHz mono float32 for ASR."""
    out = np.asarray(audio, dtype=np.float32)
    if orig_channels > 1:
        out = stereo_to_mono(out, orig_channels)
    if orig_sr != target_sr:
        out = resample(out, orig_sr, target_sr)
    return out


class AudioBuffer:
    """
    Appends float32 mono chunks at a fixed sample rate; returns a contiguous
    segment (and optionally clears it) for VAD/ASR.
    """

    def __init__(self, sample_rate: int, max_seconds: float = 30.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self._buf: list[np.ndarray] = []
        self._len = 0

    def append(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        if self._len + len(chunk) > self.max_samples:
            # Drop oldest
            keep = self.max_samples - len(chunk)
            if keep <= 0:
                self._buf = [chunk]
                self._len = len(chunk)
                return
            total = 0
            new_buf = []
            for b in reversed(self._buf):
                if total + len(b) <= keep:
                    new_buf.append(b)
                    total += len(b)
                else:
                    new_buf.append(b[-(keep - total) :])
                    total = keep
                    break
            self._buf = list(reversed(new_buf))
            self._len = sum(len(b) for b in self._buf)
        self._buf.append(chunk)
        self._len += len(chunk)

    def get_segment(self, clear: bool = True) -> np.ndarray:
        if not self._buf:
            return np.array([], dtype=np.float32)
        out = np.concatenate(self._buf)
        if clear:
            self._buf = []
            self._len = 0
        return out.astype(np.float32)

    def length_seconds(self) -> float:
        return self._len / self.sample_rate

    def clear(self) -> None:
        self._buf = []
        self._len = 0
