"""
Minimal pipeline loop: observation (mic) → buffer + VAD → ASR → action (reply).
Uses generic interface (get_observation, send_action);
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from .audio_utils import AudioBuffer, prepare_for_asr
from .vad import EnergyVAD

logger = logging.getLogger(__name__)

# Whisper often outputs these when uncertain (noisy/low-quality audio). If utterance
# is long, treat as hallucination and return empty so we don't reply with "You" etc.
_ASR_HALLUCINATION_PHRASES = frozenset({
    "you", "thank you", "thanks for watching", "bye", "thanks", "thank you for watching",
})


def _filter_asr_hallucination(text: str, duration_seconds: float) -> str:
    """Return empty string if text looks like a common Whisper hallucination for long audio."""
    if not text or duration_seconds < 4.0:
        return text
    normalized = text.strip().lower()
    if normalized in _ASR_HALLUCINATION_PHRASES:
        logger.info("ASR: treating '%s' as hallucination (%.1f s utterance)", text.strip(), duration_seconds)
        return ""
    return text


def run_pipeline_loop(
    adapter: Any,
    *,
    asr_sample_rate: int = 16000,
    mic_sample_rate: int = 48000,
    mic_channels: int = 1,
    vad_silence_duration: float = 0.8,
    vad_energy_threshold: float = 0.003,
    max_utterance_seconds: float = 30.0,
    asr: Optional[Any] = None,
    poll_interval: float = 0.05,
) -> None:
    """
    Run the pipeline loop until KeyboardInterrupt. Uses only adapter.get_observation()
    and adapter.send_action(). If asr is provided, transcribe and reply with text;
    otherwise reply with a placeholder.
    """
    buffer = AudioBuffer(asr_sample_rate, max_utterance_seconds)
    vad = EnergyVAD(
        asr_sample_rate,
        silence_duration=vad_silence_duration,
        energy_threshold=vad_energy_threshold,
    )
    while True:
        obs = adapter.get_observation()
        mic = obs.get("observation.audio.mic")

        if mic is not None and len(mic) > 0:
            chunk_16k = prepare_for_asr(
                mic,
                orig_sr=mic_sample_rate,
                orig_channels=mic_channels,
                target_sr=asr_sample_rate,
            )
            buffer.append(chunk_16k)
            if vad.process(chunk_16k):
                segment = buffer.get_segment(clear=True)
                vad.reset()
                if segment.size > 0:
                    duration_s = segment.size / asr_sample_rate
                    if asr is not None:
                        text = asr.transcribe(segment, sample_rate=asr_sample_rate)
                        text = _filter_asr_hallucination(text, duration_s)
                        logger.info("ASR: -> %s", text if text else "(empty)")
                    else:
                        text = "(no ASR)"
                        logger.info("ASR: (disabled)")
                    reply = text or "(silence)"
                    adapter.send_action({
                        "action.speech.reply_text": reply,
                        "action.speech.reply_audio": None,
                    })

        time.sleep(poll_interval)
