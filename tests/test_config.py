"""Tests for configuration module."""

import os
import pytest

from aim.config import DSPyConfig, get_config


class TestDSPyConfig:
    """Tests for DSPyConfig class."""

    def test_config_initialization_from_env(self, monkeypatch):
        """Test that config is correctly initialized from environment variables."""
        monkeypatch.setenv("AIM_OPENAI_KEY", "test-key-123")
        monkeypatch.setenv("AZURE_MODEL_NAME", "gpt-4")
        monkeypatch.setenv("AZURE_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_ENDPOINT", "https://test.openai.azure.com")

        config = DSPyConfig()

        assert config.api_key == "test-key-123"
        assert config.model_name == "gpt-4"
        assert config.api_version == "2024-02-15-preview"
        assert config.endpoint == "https://test.openai.azure.com"

    def test_config_initialization_with_missing_env(self, monkeypatch):
        """Test config initialization when environment variables are missing."""
        monkeypatch.delenv("AIM_OPENAI_KEY", raising=False)
        monkeypatch.delenv("AZURE_MODEL_NAME", raising=False)
        monkeypatch.delenv("AZURE_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_ENDPOINT", raising=False)

        config = DSPyConfig()

        assert config.api_key == ""
        assert config.model_name is None
        assert config.api_version is None
        assert config.endpoint is None

    def test_validate_with_all_config_present(self, monkeypatch):
        """Test validation passes when all required config is present."""
        monkeypatch.setenv("AIM_OPENAI_KEY", "test-key")
        monkeypatch.setenv("AZURE_MODEL_NAME", "gpt-4")
        monkeypatch.setenv("AZURE_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_ENDPOINT", "https://test.openai.azure.com")

        config = DSPyConfig()
        config.validate()  # Should not raise

    def test_validate_raises_when_api_key_missing(self, monkeypatch):
        """Test validation raises ValueError when API key is missing."""
        monkeypatch.delenv("AIM_OPENAI_KEY", raising=False)
        monkeypatch.setenv("AZURE_MODEL_NAME", "gpt-4")
        monkeypatch.setenv("AZURE_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_ENDPOINT", "https://test.openai.azure.com")

        config = DSPyConfig()

        with pytest.raises(ValueError, match="AIM_OPENAI_KEY environment variable is required"):
            config.validate()

    def test_validate_raises_when_endpoint_missing(self, monkeypatch):
        """Test validation raises ValueError when endpoint is missing."""
        monkeypatch.setenv("AIM_OPENAI_KEY", "test-key")
        monkeypatch.setenv("AZURE_MODEL_NAME", "gpt-4")
        monkeypatch.setenv("AZURE_API_VERSION", "2024-02-15-preview")
        monkeypatch.delenv("AZURE_ENDPOINT", raising=False)

        config = DSPyConfig()

        with pytest.raises(ValueError, match="AZURE_ENDPOINT environment variable is required"):
            config.validate()

    def test_validate_raises_when_model_name_missing(self, monkeypatch):
        """Test validation raises ValueError when model name is missing."""
        monkeypatch.setenv("AIM_OPENAI_KEY", "test-key")
        monkeypatch.delenv("AZURE_MODEL_NAME", raising=False)
        monkeypatch.setenv("AZURE_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_ENDPOINT", "https://test.openai.azure.com")

        config = DSPyConfig()

        with pytest.raises(ValueError, match="AZURE_MODEL_NAME environment variable is required"):
            config.validate()

    def test_validate_raises_when_api_version_missing(self, monkeypatch):
        """Test validation raises ValueError when API version is missing."""
        monkeypatch.setenv("AIM_OPENAI_KEY", "test-key")
        monkeypatch.setenv("AZURE_MODEL_NAME", "gpt-4")
        monkeypatch.delenv("AZURE_API_VERSION", raising=False)
        monkeypatch.setenv("AZURE_ENDPOINT", "https://test.openai.azure.com")

        config = DSPyConfig()

        with pytest.raises(
            ValueError, match="AZURE_API_VERSION environment variable is required"
        ):
            config.validate()

    def test_get_azure_model_path(self, monkeypatch):
        """Test Azure model path generation."""
        monkeypatch.setenv("AZURE_MODEL_NAME", "gpt-4-turbo")

        config = DSPyConfig()
        model_path = config.get_azure_model_path()

        assert model_path == "azure/gpt-4-turbo"


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_returns_validated_config(self, monkeypatch):
        """Test that get_config returns a validated DSPyConfig instance."""
        monkeypatch.setenv("AIM_OPENAI_KEY", "test-key")
        monkeypatch.setenv("AZURE_MODEL_NAME", "gpt-4")
        monkeypatch.setenv("AZURE_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_ENDPOINT", "https://test.openai.azure.com")

        config = get_config()

        assert isinstance(config, DSPyConfig)
        assert config.api_key == "test-key"
        assert config.model_name == "gpt-4"

    def test_get_config_raises_when_validation_fails(self, monkeypatch):
        """Test that get_config raises ValueError when validation fails."""
        monkeypatch.delenv("AIM_OPENAI_KEY", raising=False)
        monkeypatch.delenv("AZURE_MODEL_NAME", raising=False)
        monkeypatch.delenv("AZURE_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_ENDPOINT", raising=False)

        with pytest.raises(ValueError):
            get_config()
