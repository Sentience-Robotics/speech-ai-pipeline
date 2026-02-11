# Speech AI Pipeline

Standalone speech-to-speech pipeline: ASR (Whisper) → LLM → TTS (Piper). It connects to the robot via the generic robotics interface (LeRobot-style); no transport or middleware dependency. Use a mock adapter or a LUCY adapter (from `lucy_ros_api`) to run against a real robot.

## Generic interface

This package uses only a generic observation/action interface: `get_observation()` and `send_action(action)` with the speech subset of keys. The interface and its adapters are defined and implemented in **`lucy_ros_api`** (LUCY workspace).

- **Contract:** `get_observation()` returns e.g. `observation.audio.mic`, `observation.speech.user_transcript`; `send_action(action)` accepts `action.speech.reply_text` or `action.speech.reply_audio`.
- **Documentation:** See `lucy_ros_api` docs (e.g. `docs/GENERIC_INTERFACE.md`) and Python package `lucy_ros_api.generic_interface`.

### Run (mock or LUCY)

Put `lucy_ros_api` on PYTHONPATH (e.g. build and source the LUCY workspace), then from this directory:

```bash
# Optional: build and source LUCY workspace for lucy_ros_api
# cd ~/lucy_ws && colcon build --packages-select lucy_ros_api && source install/setup.bash
# cd ~/speech_ai_pipeline

python run_with_generic_interface.py --mock   # Mock adapter
python run_with_generic_interface.py --lucy  # LUCY adapter (LUCY must be reachable)
```

## Project layout

- `config/config.example.yaml` – Example config (copy to `config.yaml`).
- `run_with_generic_interface.py` – Entry point using the generic interface only.
