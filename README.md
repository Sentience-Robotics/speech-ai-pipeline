# Speech AI Pipeline

Standalone speech-to-speech pipeline: ASR (Whisper) → LLM → TTS (Piper). It connects to the robot via the generic robotics interface (LeRobot-style); no transport or middleware dependency. Use a mock adapter or a LUCY adapter (from `lucy_ros_api`) to run against a real robot.

## Generic interface

This package uses only a generic observation/action interface: `get_observation()` and `send_action(action)` with the speech subset of keys. The interface and its adapters are defined and implemented in **`lucy_ros_api`** (LUCY workspace).

- **Contract:** `get_observation()` returns e.g. `observation.audio.mic`, `observation.speech.user_transcript`; `send_action(action)` accepts `action.speech.reply_text` or `action.speech.reply_audio`.
- **Documentation:** See `lucy_ros_api` docs (e.g. `docs/GENERIC_INTERFACE.md`) and Python package `lucy_ros_api.generic_interface`.

### Running with a ROS backend (e.g. LUCY)

You can run the pipeline with a ROS-based backend **if**:

1. The robot/stack that provides the generic interface is running (e.g. LUCY with audio enabled).
2. The backend publishes mic data and subscribes to playback on the topics expected by the adapter (see the backend’s interface docs).
3. The adapter package (e.g. `lucy_ros_api`) is on PYTHONPATH.

Then from this directory: `python run_pipeline.py --lucy` (or `--lucy --no-asr` to skip ASR). Without a running backend, use `--mock` for adapter and pipeline tests.

### Run

Put `lucy_ros_api` on PYTHONPATH (e.g. build and source the LUCY workspace), then from this directory:

```bash
# Quick test (adapter contract only)
python run_with_generic_interface.py --mock
python run_with_generic_interface.py --lucy

# Full pipeline: mic → VAD → ASR → reply (via /say or mock)
python run_pipeline.py --lucy              # With ASR (faster-whisper)
python run_pipeline.py --lucy --no-asr     # Placeholder reply only
python run_pipeline.py --mock              # Mock adapter (no real mic)
```

Config: copy `config/config.example.yaml` to `config/config.yaml` and adjust (ASR model, VAD, etc.).

## GPU (CUDA) for ASR

ASR uses **faster-whisper**, which depends on **CTranslate2**. The error *"This CTranslate2 package was not compiled with CUDA support"* means the installed `ctranslate2` wheel is CPU-only.

- **x86_64 Linux:** The default PyPI wheel usually includes CUDA. Install CUDA 12.x (and cuDNN 8 for speech models) on the system, then `pip install ctranslate2` and set `asr.device: "cuda"` in config.
- **Jetson (aarch64):** Many installs get a CPU-only wheel (e.g. from piwheels or a generic aarch64 build). To use the GPU you need a CTranslate2 build with CUDA:
  1. Install CUDA and cuDNN (e.g. via JetPack).
  2. Build CTranslate2 from source with CUDA enabled:
     ```bash
     git clone --recursive https://github.com/OpenNMT/CTranslate2.git && cd CTranslate2
     mkdir build && cd build
     cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCMAKE_INSTALL_PREFIX=/usr/local
     make -j$(nproc) && sudo make install && sudo ldconfig
     cd ../python && pip install -r install_requirements.txt && python setup.py bdist_wheel && pip install dist/*.whl
     ```
  3. Keep `asr.device: "cuda"` in `config.yaml`.

If you don’t need GPU, leave the CPU fallback (or set `asr.device: "cpu"`); the pipeline will log a warning and use CPU.

## Tests

```bash
python3 -m pytest
```

With coverage (report in terminal and `htmlcov/`):

```bash
python3 -m pytest --cov=src --cov-report=term-missing --cov-report=html --cov-branch
```

Requires the generic-interface package (e.g. `lucy_ros_api`) on PYTHONPATH for `tests/test_pipeline.py` (or that test is skipped). Add `htmlcov/` to `.gitignore` if you generate reports.

## Integration testing (generic interface)

The pipeline talks to the robot only via the generic interface (`get_observation` / `send_action`). Integration testing means running the pipeline against a real backend that implements that interface—not a specific robot or repo.

- **Mock adapter** (`--mock`): No hardware or ROS. Use for quick sanity checks and CI.
- **ROS adapter** (e.g. `--lucy`): Use when a ROS graph is running that provides the interface (e.g. mic topic and playback topic). The backend’s docs describe how to start and verify that graph; this repo only needs the adapter on PYTHONPATH and the backend running.

Example with a mock (no external stack):

```bash
python run_pipeline.py --mock
```

Example with a ROS-based backend: source the workspace that provides the adapter, start the robot/audio stack as documented there, then from this repo run e.g. `python run_pipeline.py --lucy`. Exact launch commands and topic names are defined by the backend, not here.

## Project layout

- `config/config.example.yaml` – Example config (copy to `config.yaml`).
- `run_with_generic_interface.py` – Adapter test (no pipeline).
- `run_pipeline.py` – Pipeline loop: observation → buffer + VAD → ASR → send_action.
- `src/` – `audio_utils`, `vad`, `asr`, `config_loader`, `pipeline`.
- `tests/` – Unit tests (pytest).
