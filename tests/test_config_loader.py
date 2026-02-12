"""Unit tests for config_loader."""
import pytest
from src.config_loader import load_config, _default_config


def test_load_config_returns_dict():
    cfg = load_config()
    assert isinstance(cfg, dict)


def test_default_config_has_pipeline():
    cfg = _default_config()
    assert "pipeline" in cfg
    assert "asr_sample_rate" in cfg["pipeline"] or "asr" in cfg


def test_load_config_has_expected_keys():
    cfg = load_config()
    assert "pipeline" in cfg
    assert cfg["pipeline"].get("asr_sample_rate") == 16000
