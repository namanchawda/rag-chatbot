from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    DATABASE_URL: str
    JWT_SECRET: str
    JWT_EXPIRE_MINUTES: int = 60
    FRONTEND_ORIGIN: str = "http://localhost:3000"
    JINA_API_KEY: str = ""
    JINA_EMBEDDING_MODEL: str = "jina-embeddings-v5-omni-small"
    EMBEDDING_DIMENSION: int = 1024
    MAX_UPLOAD_MB: int = 25
    MAX_PDF_PAGES: int = 300
    MAX_CHUNKS_PER_DOCUMENT: int = 1500
    MAX_TOTAL_CHUNKS_PER_USER: int = 3000
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    # Selectable chunking strategies (see rag/chunking.py).
    CHUNK_SIZE_MIN: int = 300
    CHUNK_SIZE_MAX: int = 2000
    CHUNK_OVERLAP_MAX: int = 300
    DEFAULT_CHUNKING_STRATEGY: str = "recursive"
    GROQ_API_KEY: str
    GROQ_MODEL: str
    RETRIEVAL_TOP_K: int = 5
    MAX_QUESTION_CHARS: int = 2000
    RERANK_ENABLED: bool = True
    JINA_RERANK_MODEL: str = "jina-reranker-v3.5"
    RETRIEVAL_CANDIDATES: int = 20
    RERANK_TOP_N: int = 5
    RERANK_MIN_SCORE: float | None = None
    RERANK_LOG_SCORES: bool = False
    LOG_LEVEL: str = "INFO"
    MAX_DOCUMENTS_PER_USER: int = 10
    MAX_MESSAGES_PER_DAY: int = 50
    STALE_PROCESSING_MINUTES: int = 30
    AUTH_RATE_LIMIT_PER_MINUTE: int = 10
    SIGNUPS_ENABLED: bool = True
    TRUST_PROXY: bool = False
    FRONTEND_ORIGINS: str = ""
    # Admin status API. Comma-separated emails, matched case-insensitively;
    # empty means nobody is an admin.
    ADMIN_EMAILS: str = ""
    # Balance shown on the Jina dashboard when the key was installed.
    JINA_TOKEN_BUDGET: int = 10000000
    # Low-balance flag threshold: percent of the budget remaining.
    JINA_LOW_BALANCE_PERCENT: int = 15
    # Minimum seconds between /admin/status/check calls.
    STATUS_CHECK_COOLDOWN_SECONDS: int = 30
    EMAIL_BACKEND: str = "console"
    BREVO_API_KEY: str = ""
    EMAIL_FROM_ADDRESS: str = ""
    EMAIL_FROM_NAME: str = "RAG Chatbot"
    GMAIL_CLIENT_ID: str = ""
    GMAIL_CLIENT_SECRET: str = ""
    GMAIL_REFRESH_TOKEN: str = ""
    GMAIL_USER: str = ""
    CODE_TTL_MINUTES: int = 15
    CODE_MAX_ATTEMPTS: int = 5
    EMAIL_RESEND_COOLDOWN_SECONDS: int = 60
    MAX_EMAILS_PER_DAY: int = 400
    UNVERIFIED_ACCOUNT_TTL_HOURS: int = 24

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    @field_validator("EMAIL_BACKEND")
    @classmethod
    def validate_email_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"console", "brevo", "gmail"}:
            raise ValueError("EMAIL_BACKEND must be either 'console', 'brevo', or 'gmail'")
        return normalized


def validate_email_settings(settings: Settings | None = None) -> None:
    """Fail startup clearly when email backend is selected without its credentials."""
    current = settings or get_settings()
    if current.EMAIL_BACKEND == "brevo":
        missing = []
        if not current.BREVO_API_KEY.strip():
            missing.append("BREVO_API_KEY")
        if not current.EMAIL_FROM_ADDRESS.strip():
            missing.append("EMAIL_FROM_ADDRESS")
        if missing:
            raise RuntimeError(
                "EMAIL_BACKEND=brevo requires " + " and ".join(missing) + "."
            )
    elif current.EMAIL_BACKEND == "gmail":
        missing = []
        if not current.GMAIL_CLIENT_ID.strip():
            missing.append("GMAIL_CLIENT_ID")
        if not current.GMAIL_CLIENT_SECRET.strip():
            missing.append("GMAIL_CLIENT_SECRET")
        if not current.GMAIL_REFRESH_TOKEN.strip():
            missing.append("GMAIL_REFRESH_TOKEN")
        if not current.GMAIL_USER.strip():
            missing.append("GMAIL_USER")
        if missing:
            raise RuntimeError(
                "EMAIL_BACKEND=gmail requires " + " and ".join(missing) + "."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
