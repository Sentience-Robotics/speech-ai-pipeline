"""
Load pipeline config from config.yaml (or config.example.yaml or env).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def _find_config_dir() -> Path:
    """Config next to repo root (parent of src/ or cwd)."""
    for d in [Path.cwd(), Path(__file__).resolve().parent.parent]:
        if (d / "config" / "config.yaml").exists():
            return d / "config"
        if (d / "config" / "config.example.yaml").exists():
            return d / "config"
    return Path.cwd() / "config"


def load_config() -> dict[str, Any]:
    """Load config from config.yaml or config.example.yaml."""
    if not yaml:
        return _default_config()
    config_dir = _find_config_dir()
    for name in ("config.yaml", "config.example.yaml"):
        path = config_dir / name
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else _default_config()
    return _default_config()


def _default_config() -> dict[str, Any]:
    return {
        "pipeline": {
            "asr_sample_rate": 16000,
            "asr_channels": 1,
            "vad_silence_duration": 0.8,
            "max_utterance_seconds": 30.0,
        },
        "asr": {
            "backend": "faster_whisper",
            "model_size": "small",
            "device": "cpu",
            "language": None,
        },
        "connection": {
            "playback_rate": 48000,
            "playback_channels": 1,
            "playback_chunk_size": 1024,
        },
    }
