from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import create_tables
from .rag.ingest import ensure_pgvector
from .routers import auth, documents


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

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return api


app = create_app()
