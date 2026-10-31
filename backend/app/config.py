from functools import lru_cache
from pathlib import Path

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

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
