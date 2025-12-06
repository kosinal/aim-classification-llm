from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Configuration for application"""
    model_base_path: str

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()