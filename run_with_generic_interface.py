#!/usr/bin/env python3
"""
Run using only the generic robot interface (LeRobot-style).

Uses get_observation() and send_action() with the speech subset. No transport
or middleware dependency; the adapter (mock or LUCY) is chosen at run time.

Usage:
  python run_with_generic_interface.py --mock   # Mock adapter (no robot)
  python run_with_generic_interface.py --lucy  # LUCY adapter (requires LUCY reachable via lucy_ros_api)
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run via generic interface")
    parser.add_argument("--mock", action="store_true", help="Use mock adapter")
    parser.add_argument("--lucy", action="store_true", help="Use LUCY adapter")
    args = parser.parse_args()

    if not args.mock and not args.lucy:
        parser.error("Choose --mock or --lucy")

    if args.mock:
        from lucy_ros_api.generic_interface import MockSpeechAdapter
        adapter = MockSpeechAdapter()
        adapter.set_transcript("Hello, robot!")
        # Simulate one turn: pipeline would run ASR → LLM → TTS; here we just echo
        obs = adapter.get_observation()
        assert "observation.speech.user_transcript" in obs
        assert "observation.audio.mic" in obs
        # Send a reply (e.g. from TTS output)
        adapter.send_action({
            "action.speech.reply_text": "Hello! How can I help?",
            "action.speech.reply_audio": None,
        })
        print("Mock: last reply text =", adapter.get_last_reply_text())
        print("Generic interface (mock) OK.")
        return 0

    if args.lucy:
        from lucy_ros_api.generic_interface import LucyROSAdapter
        adapter = LucyROSAdapter(
            mic_topic="/mic_audio",
            playback_topic="/audio",
            say_action="/say",
            playback_rate=48000,
            playback_channels=1,
            playback_chunk_size=1024,
        )
        adapter.start()
        try:
            obs = adapter.get_observation()
            print("Observation keys:", list(obs.keys()))
            adapter.send_action({"action.speech.reply_text": "Generic interface test.", "action.speech.reply_audio": None})
            print("Generic interface (LUCY) OK. Reply sent.")
        finally:
            adapter.stop()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
