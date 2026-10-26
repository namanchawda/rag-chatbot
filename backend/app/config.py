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
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
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

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
