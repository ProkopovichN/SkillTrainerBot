import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


load_dotenv()


def _getenv(name: str) -> str:
    return os.getenv(name, "").strip()


@dataclass
class Settings:
    app_host: str = field(default_factory=lambda: _getenv("BACKEND_HOST") or "0.0.0.0")
    app_port: int = field(default_factory=lambda: int(_getenv("BACKEND_PORT") or 8000))
    frontend_push_url: str | None = field(
        default_factory=lambda: _getenv("FRONTEND_PUSH_URL") or None
    )
    frontend_push_token: str | None = field(
        default_factory=lambda: _getenv("FRONTEND_PUSH_TOKEN") or None
    )
    reminder_delay_seconds: int = field(
        default_factory=lambda: int(_getenv("REMINDER_DELAY_SECONDS") or 300)
    )
    openrouter_api_key: str | None = field(
        default_factory=lambda: _getenv("OPENROUTER_API_KEY") or None
    )
    openrouter_model: str = field(
        default_factory=lambda: _getenv("OPENROUTER_MODEL") or "gpt-3.5-turbo"
    )
    openrouter_base_url: str = field(
        default_factory=lambda: _getenv("OPENROUTER_BASE_URL")
        or "https://api.openai.com/v1/chat/completions"
    )
    openrouter_temperature: float = field(
        default_factory=lambda: float(_getenv("OPENROUTER_TEMPERATURE") or 0.2)
    )
    ai_positive_keywords: tuple[str, ...] = (
        "конструктив",
        "конкретно",
        "действия",
        "пример",
        "ожидания",
    )
