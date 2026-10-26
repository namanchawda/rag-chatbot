import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import create_tables
from .rag.ingest import ensure_pgvector
from .routers import auth, chat, documents, sessions

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
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    api = FastAPI(title="RAG Chatbot API", lifespan=lifespan)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api.include_router(auth.router)
    api.include_router(documents.router)
    api.include_router(chat.router)
    api.include_router(sessions.router)

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return api


app = create_app()
