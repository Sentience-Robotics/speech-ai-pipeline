"""Unit tests for audio_utils."""
import numpy as np
import pytest
from src.audio_utils import (
    resample,
    stereo_to_mono,
    prepare_for_asr,
    AudioBuffer,
)


def test_resample_same_rate():
    x = np.random.randn(160).astype(np.float32)
    out = resample(x, 16000, 16000)
    np.testing.assert_array_almost_equal(out, x)


def test_resample_down():
    x = np.random.randn(480).astype(np.float32)
    out = resample(x, 48000, 16000)
    assert out.dtype == np.float32
    assert len(out) == 160


def test_resample_up():
    x = np.random.randn(160).astype(np.float32)
    out = resample(x, 16000, 48000)
    assert out.dtype == np.float32
    assert len(out) == 480


def test_stereo_to_mono_interleaved():
    # 4 samples stereo = 8 values
    x = np.array([1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0], dtype=np.float32)
    out = stereo_to_mono(x, channels=2)
    assert out.shape == (4,)
    np.testing.assert_array_almost_equal(out, [0.5, 0.5, -0.5, -0.5])


def test_prepare_for_asr_stereo_48k():
    x = np.random.randn(960).astype(np.float32)  # 10 ms at 48k stereo = 480 samples per channel
    out = prepare_for_asr(x, orig_sr=48000, orig_channels=2, target_sr=16000)
    assert out.dtype == np.float32
    assert len(out) == 160  # 10 ms at 16k


def test_audio_buffer_append_get():
    buf = AudioBuffer(sample_rate=16000, max_seconds=1.0)
    buf.append(np.ones(160, dtype=np.float32))
    buf.append(np.ones(160, dtype=np.float32))
    seg = buf.get_segment(clear=True)
    assert len(seg) == 320
    assert buf.length_seconds() == 0


def test_audio_buffer_clear():
    buf = AudioBuffer(sample_rate=16000, max_seconds=1.0)
    buf.append(np.ones(160, dtype=np.float32))
    buf.clear()
    seg = buf.get_segment(clear=True)
    assert len(seg) == 0
