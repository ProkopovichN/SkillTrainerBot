import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


load_dotenv()


def _getenv(name: str) -> str:
    return os.getenv(name, "").strip()


@dataclass
class Settings:
    bot_token: str = field(default_factory=lambda: _getenv("BOT_TOKEN"))
    backend_url: str = field(default_factory=lambda: _getenv("BACKEND_URL"))
    backend_token: str | None = field(
        default_factory=lambda: _getenv("BACKEND_TOKEN") or None
    )
    push_token: str | None = field(
        default_factory=lambda: _getenv("PUSH_TOKEN") or None
    )
    webhook_url: str = field(default_factory=lambda: _getenv("WEBHOOK_URL"))
    webhook_path: str = field(
        default_factory=lambda: _getenv("WEBHOOK_PATH") or "/tg/webhook"
    )
    webhook_secret: str | None = field(
        default_factory=lambda: _getenv("WEBHOOK_SECRET") or None
    )
    use_webhook: bool = field(
        default_factory=lambda: (_getenv("USE_WEBHOOK") or "true").lower() == "true"
    )
    listen_host: str = field(default_factory=lambda: _getenv("LISTEN_HOST") or "0.0.0.0")
    listen_port: int = field(
        default_factory=lambda: int(_getenv("LISTEN_PORT") or 8080)
    )
    transcribe_url: str | None = field(
        default_factory=lambda: _getenv("TRANSCRIBE_URL") or None
    )
    transcribe_token: str | None = field(
        default_factory=lambda: _getenv("TRANSCRIBE_TOKEN") or None
    )
    openrouter_api_key: str | None = field(
        default_factory=lambda: _getenv("OPENROUTER_API_KEY") or None
    )
    openrouter_asr_model: str | None = field(
        default_factory=lambda: _getenv("OPENROUTER_ASR_MODEL") or None
    )
    openrouter_asr_chat_model: str | None = field(
        default_factory=lambda: _getenv("OPENROUTER_ASR_CHAT_MODEL") or None
    )
    openrouter_asr_url: str = field(
        default_factory=lambda: _getenv("OPENROUTER_ASR_URL")
        or "https://openrouter.ai/api/v1/chat/completions"
    )
    deepgram_api_key: str | None = field(
        default_factory=lambda: _getenv("DEEPGRAM_API_KEY") or None
    )
    deepgram_url: str = field(
        default_factory=lambda: _getenv("DEEPGRAM_URL") or "https://api.deepgram.com/v1/listen"
    )
    deepgram_model: str | None = field(
        default_factory=lambda: _getenv("DEEPGRAM_MODEL") or None
    )
    deepgram_language: str | None = field(
        default_factory=lambda: _getenv("DEEPGRAM_LANGUAGE") or None
    )
    default_reply_text: str = field(
        default_factory=lambda: _getenv("DEFAULT_REPLY_TEXT")
        or "Привет! Я помогу потренировать навыки ревью и планов развития. Пришли текст или голос."
    )
    log_level: str = field(
        default_factory=lambda: (_getenv("LOG_LEVEL") or "INFO").upper()
    )
    request_timeout_seconds: float = field(
        default_factory=lambda: float(_getenv("REQUEST_TIMEOUT_SECONDS") or 15.0)
    )
    asr_timeout_seconds: float = field(
        default_factory=lambda: float(_getenv("ASR_TIMEOUT_SECONDS") or 20.0)
    )
    ffmpeg_binary: str = field(
        default_factory=lambda: _getenv("FFMPEG_BINARY") or "ffmpeg"
    )
    max_tg_message_length: int = field(
        default_factory=lambda: int(_getenv("MAX_TG_MESSAGE_LENGTH") or 3900)
    )

    def __post_init__(self) -> None:
        if not self.bot_token:
            raise ValueError("BOT_TOKEN is required")
        if not self.backend_url:
            raise ValueError("BACKEND_URL is required")
        if self.use_webhook and not self.webhook_url:
            raise ValueError("WEBHOOK_URL is required for webhook mode")
