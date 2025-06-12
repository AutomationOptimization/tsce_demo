from functools import lru_cache
from pydantic import ValidationError
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Central runtime configuration loaded from environment variables."""
    openai_key: str
    openai_endpoint: str | None = None
    model_name: str = "gpt-3.5-turbo"
    log_dir: str = "logs"

    class Config:
        env_file = ".env"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    try:
        return Settings()
    except ValidationError as exc:
        msg = (
            "Missing required configuration. Ensure OPENAI_KEY is set in your"
            " environment or .env file."
        )
        raise EnvironmentError(msg) from exc
