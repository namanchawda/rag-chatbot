import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
from .db import create_tables
from .maintenance import fail_stale_uploads
from .rag.ingest import ensure_pgvector
from .routers import auth, chat, chunks, documents, sessions

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

_GUARD_FLAG = "_rag_chatbot_stream_handler"


def configure_logging() -> None:
    """Attach one StreamHandler to the "app" logger once.

    Uvicorn only configures its own loggers, so without this, logger.info
    calls from app.* fall through to the root handler's WARNING default and
    are dropped. propagate=False keeps records from reaching it twice.
    """
    app_logger = logging.getLogger("app")
    if any(getattr(handler, _GUARD_FLAG, False) for handler in app_logger.handlers):
        return
    settings = get_settings()
    requested = (settings.LOG_LEVEL or "INFO").strip().upper()
    level = logging.getLevelName(requested)
    if not isinstance(level, int):
        level = logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    setattr(handler, _GUARD_FLAG, True)
    handler.setLevel(logging.DEBUG)
    app_logger.setLevel(level)
    app_logger.addHandler(handler)
    app_logger.propagate = False


configure_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_pgvector()
    create_tables()
    try:
        fail_stale_uploads()
    except Exception as error:  # noqa: BLE001 - startup must survive a sweep failure
        logging.getLogger(__name__).warning(
            "Startup upload sweep failed (%s); the app will still start",
            type(error).__name__,
        )
    yield


def allowed_origins(settings: Settings) -> list[str]:
    """Exact origins only - a rogue wildcard would open the API to any site."""
    listed = [origin.strip() for origin in settings.FRONTEND_ORIGINS.split(",")]
    origins = [origin for origin in listed if origin and origin != "*"]
    if not origins:
        fallback = settings.FRONTEND_ORIGIN.strip()
        if fallback and fallback != "*":
            origins = [fallback]
    return origins


def create_app() -> FastAPI:
    settings = get_settings()
    api = FastAPI(title="RAG Chatbot API", lifespan=lifespan)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(settings),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api.include_router(auth.router)
    api.include_router(documents.router)
    api.include_router(chat.router)
    api.include_router(sessions.router)
    api.include_router(chunks.router)

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/config")
    def public_config() -> dict:
        settings = get_settings()
        return {
            "signups_enabled": bool(settings.SIGNUPS_ENABLED),
            "max_upload_mb": settings.MAX_UPLOAD_MB,
            "allowed_extensions": [".pdf"],
            "rerank_available": bool(settings.RERANK_ENABLED),
            "rerank_default": bool(settings.RERANK_ENABLED),
        }

    return api


app = create_app()
