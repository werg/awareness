"""Tests for configuration module."""

import pytest
from awareness.config import Config, EncoderConfig, DecoderConfig


def test_config_creation():
    """Test that Config can be created with defaults."""
    config = Config()
    assert config is not None
    assert config.device in ["cuda", "cpu"]


def test_encoder_config():
    """Test EncoderConfig."""
    config = EncoderConfig()
    assert config.hidden_size == 1024
    assert config.num_attention_heads == 16


def test_decoder_config():
    """Test DecoderConfig."""
    config = DecoderConfig()
    assert config.gca_enabled is True
    assert config.gca_start_layer == 18


def test_config_to_dict():
    """Test config serialization to dict."""
    config = Config()
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "encoder" in config_dict
    assert "decoder" in config_dict
    assert "training" in config_dict
