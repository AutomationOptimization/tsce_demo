from functools import lru_cache
import os

try:
    from pydantic import ValidationError  # type: ignore
    from pydantic_settings import BaseSettings  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ValidationError = Exception

class Settings:
    """Minimal runtime configuration loaded from environment variables."""

    def __init__(self) -> None:
        self.openai_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("OPENAI_ENDPOINT")
        self.model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self.log_dir = os.getenv("LOG_DIR", "logs")
        if not self.openai_key:
            msg = (
                "Missing required configuration. Ensure OPENAI_KEY is set in your"
                " environment or .env file."
            )
            raise EnvironmentError(msg)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
