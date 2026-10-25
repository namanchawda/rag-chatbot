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

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
