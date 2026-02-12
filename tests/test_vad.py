"""Unit tests for VAD."""
import numpy as np
import pytest
from src.vad import rms, EnergyVAD


def test_rms_empty():
    assert rms(np.array([], dtype=np.float32)) == 0.0


def test_rms_silence():
    assert rms(np.zeros(100, dtype=np.float32)) == 0.0


def test_rms_signal():
    x = np.ones(100, dtype=np.float32)
    assert abs(rms(x) - 1.0) < 1e-6


def test_energy_vad_silence_only():
    vad = EnergyVAD(sample_rate=16000, silence_duration=0.2, energy_threshold=0.01)
    silence = np.zeros(1600, dtype=np.float32)  # 0.1 s
    assert vad.process(silence) is False
    assert vad.process(silence) is False


def test_energy_vad_speech_then_silence():
    vad = EnergyVAD(sample_rate=16000, silence_duration=0.2, energy_threshold=0.01, frame_seconds=0.05)
    vad._silence_frames_needed = 4
    speech = np.ones(800, dtype=np.float32) * 0.5
    silence = np.zeros(800, dtype=np.float32)
    vad.process(speech)
    triggered = False
    for _ in range(10):
        if vad.process(silence):
            triggered = True
            break
    assert triggered
