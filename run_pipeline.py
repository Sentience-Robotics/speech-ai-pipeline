#!/usr/bin/env python3
"""
Run the speech pipeline: observation → buffer + VAD → ASR → reply via generic interface.

Usage:
  python run_pipeline.py --mock [--no-asr]   # Mock adapter; --no-asr echoes placeholder
  python run_pipeline.py --lucy [--no-asr]   # LUCY adapter (requires lucy_ros_api on PYTHONPATH)
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Add src for imports when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config_loader import load_config
from src.pipeline import run_pipeline_loop
from src.asr import ASR


def main() -> int:
    parser = argparse.ArgumentParser(description="Run speech pipeline via generic interface")
    parser.add_argument("--mock", action="store_true", help="Use mock adapter")
    parser.add_argument("--lucy", action="store_true", help="Use LUCY adapter")
    parser.add_argument("--no-asr", action="store_true", help="Skip ASR; reply with placeholder")
    args = parser.parse_args()

    if not args.mock and not args.lucy:
        parser.error("Choose --mock or --lucy")

    config = load_config()
    pipe_cfg = config.get("pipeline", {})
    asr_cfg = config.get("asr", {})
    conn_cfg = config.get("connection", {})

    if args.mock:
        from lucy_ros_api.generic_interface import MockSpeechAdapter
        adapter = MockSpeechAdapter()
        # For mock, feed mic in the loop from pipeline; no real mic. So we need to
        # either inject audio or run with --lucy for real mic. For mock we can run
        # a one-shot: set_transcript, get_observation, then pipeline would need to
        # use transcript when mic is None. For simplicity: mock with no mic chunks
        # will never trigger VAD; so run with --lucy for real pipeline test, or
        # add a mock that yields fake mic. Here we just start the loop; with mock
        # and no set_mic_chunk, obs.audio.mic will be None so nothing is buffered.
        # So for mock we could set periodic fake mic - or document that --mock is
        # for testing adapter contract and --lucy for full pipeline.
    else:
        from lucy_ros_api.generic_interface import LucyROSAdapter
        adapter = LucyROSAdapter(
            mic_topic=conn_cfg.get("mic_topic", "/mic_audio"),
            playback_topic=conn_cfg.get("playback_topic", "/audio"),
            say_action=conn_cfg.get("say_action", "/say"),
            playback_rate=conn_cfg.get("playback_rate", 48000),
            playback_channels=conn_cfg.get("playback_channels", 1),
            playback_chunk_size=conn_cfg.get("playback_chunk_size", 1024),
        )
        adapter.start()

    asr = None
    if not args.no_asr:
        asr = ASR(
            backend=asr_cfg.get("backend", "faster_whisper"),
            model_size=asr_cfg.get("model_size", "small"),
            device=asr_cfg.get("device", "cpu"),
            language=asr_cfg.get("language"),
            model_path=asr_cfg.get("model_path"),
            no_speech_threshold=asr_cfg.get("no_speech_threshold", 0.4),
        )

    try:
        run_pipeline_loop(
            adapter,
            asr_sample_rate=pipe_cfg.get("asr_sample_rate", 16000),
            mic_sample_rate=conn_cfg.get("mic_sample_rate", 48000),
            mic_channels=pipe_cfg.get("asr_channels", 1),
            vad_silence_duration=pipe_cfg.get("vad_silence_duration", 0.8),
            vad_energy_threshold=pipe_cfg.get("vad_energy_threshold", 0.003),
            max_utterance_seconds=pipe_cfg.get("max_utterance_seconds", 30.0),
            asr=asr,
            poll_interval=0.05,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if args.lucy and hasattr(adapter, "stop"):
            adapter.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
