from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and `.env`.

    Secrets are represented as `SecretStr` so they are never printed accidentally.
    """

    app_name: str = "FitLife AI"
    app_log_level: str = "INFO"

    llm_api_key: SecretStr = Field(alias="LLM_API_KEY")
    llm_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="LLM_BASE_URL",
    )
    llm_model: str = Field(default="qwen-plus", alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=60.0, gt=0, alias="LLM_TIMEOUT_SECONDS")
    llm_temperature: float = Field(default=0.2, ge=0, le=2, alias="LLM_TEMPERATURE")

    database_url: str = Field(default="sqlite:///./fitlife.db", alias="DATABASE_URL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("database_url")
    @classmethod
    def validate_sqlite_database_url(cls, value: str) -> str:
        if not value.startswith("sqlite:"):
            raise ValueError("Phase 2.1 requires a SQLite DATABASE_URL")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()