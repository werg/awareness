"""Tests for configuration module."""

from awareness.config import Config, EncoderConfig, DecoderConfig


def test_config_creation():
    """Test that Config can be created with defaults."""
    config = Config()
    assert config is not None
    assert config.device == "cuda"


def test_encoder_config():
    """Test EncoderConfig can be instantiated."""
    config = EncoderConfig()
    assert config.model_name is None  # No default - to be chosen


def test_decoder_config():
    """Test DecoderConfig can be instantiated."""
    config = DecoderConfig()
    assert config.model_name is None  # No default - to be chosen
