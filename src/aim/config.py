"""Configuration module for DSPy and Azure OpenAI settings."""

import os

class DSPyConfig:
    """Configuration for DSPy Language Model using Azure OpenAI."""

    def __init__(self) -> None:
        """Initialize configuration from environment variables."""
        self.api_key: str = os.environ.get("AIM_OPENAI_KEY", "")
        self.model_name: str = os.environ.get("AZURE_MODEL_NAME")
        self.api_version: str = os.environ.get("AZURE_API_VERSION")
        self.endpoint: str = os.environ.get("AZURE_ENDPOINT")

    def validate(self) -> None:
        """
        Validate that all required configuration is present.

        Raises:
            ValueError: If required configuration is missing
        """
        check_pairs = [
            (self.api_key, "AIM_OPENAI_KEY"),
            (self.endpoint, "AZURE_ENDPOINT"),
            (self.model_name, "AZURE_MODEL_NAME"),
            (self.api_version, "AZURE_API_VERSION"),
        ]
        for config_var, env_name in check_pairs:
            if not config_var:
                raise ValueError(f"{env_name} environment variable is required")

    def get_azure_model_path(self) -> str:
        """
        Get the full Azure model path for DSPy.

        Returns:
            Full model path in format "azure/{model_name}"
        """
        return f"azure/{self.model_name}"


def get_config() -> DSPyConfig:
    """
    Get validated DSPy configuration from environment variables.

    Returns:
        DSPyConfig instance with validated settings

    Raises:
        ValueError: If required configuration is missing
    """
    config = DSPyConfig()
    config.validate()
    return config
