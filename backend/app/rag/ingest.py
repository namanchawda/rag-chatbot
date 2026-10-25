"""Document loading, chunking, and persistence into pgvector."""

import logging
import uuid as uuid_module
from collections.abc import Callable
from pathlib import Path

from langchain_community.document_loaders import BSHTMLLoader, PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine

from ..config import get_settings
from ..db import SessionLocal, engine
from ..models import Document as DocumentRow
from ..models import DocumentStatus
from .embeddings import Embeddings, JinaEmbeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag-chatbot-chunks"
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".html"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file suffix is not in the supported set."""


def ensure_pgvector(target_engine: Engine | None = None) -> None:
    """Create the pgvector extension, once, at application startup."""
    target_engine = target_engine if target_engine is not None else engine
    if target_engine.url.drivername.startswith("sqlite"):
        return
    with target_engine.connect() as connection:
        connection.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.commit()
    logger.info("pgvector extension ensured")


def build_vectorstore(embeddings: Embeddings | None = None) -> PGVector:
    """Create the shared pgvector-backed store in one fixed collection.

    The vector dimension comes exclusively from EMBEDDING_DIMENSION.
    """
    settings = get_settings()
    return PGVector(
        embeddings=embeddings if embeddings is not None else JinaEmbeddings(),
        connection=settings.DATABASE_URL,
        collection_name=COLLECTION_NAME,
        collection_metadata={"app": "rag-chatbot"},
        embedding_length=settings.EMBEDDING_DIMENSION,
        distance_strategy=DistanceStrategy.COSINE,
        engine_args={"pool_pre_ping": True, "pool_recycle": 300},
        create_extension=False,
    )


def load_document(temp_path: str | Path, filename: str) -> list[Document]:
    """Load a single temporary file, dispatched strictly by suffix."""
    suffix = Path(filename).suffix.casefold()
    resolved = str(temp_path)
    if suffix == ".pdf":
        return PyMuPDFLoader(resolved, mode="single").load()
    if suffix in (".txt", ".md"):
        try:
            return TextLoader(resolved, encoding="utf-8").load()
        except (UnicodeDecodeError, RuntimeError):
            # TextLoader wraps decode failures in a RuntimeError; retry with a
            # permissive encoding before giving up.
            return TextLoader(resolved, encoding="latin-1").load()
    if suffix == ".html":
        return BSHTMLLoader(resolved, bs_kwargs={"features": "html.parser"}).load()
    raise UnsupportedFileTypeError(
        f"Unsupported file type '{suffix}'. Supported types: .pdf, .txt, .md, .html"
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split loaded documents with the configured size and overlap."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)


def delete_document_vectors(store: PGVector, document_id: str, chunk_count: int) -> None:
    """Remove every vector that belongs to one document, by id."""
    ids = [f"{document_id}:{index}" for index in range(chunk_count)]
    store.delete(ids=ids)


def _annotate(
    chunk: Document,
    *,
    user_id: str,
    document_id: str,
    source: str,
    index: int,
) -> Document:
    chunk.metadata.update(
        {
            "user_id": user_id,
            "document_id": document_id,
            "source": source,
            "chunk_index": index,
        }
    )
    chunk.id = f"{document_id}:{index}"
    return chunk


def ingest_document(
    document_id: str,
    temp_path: str | Path,
    filename: str,
    user_id: str,
    *,
    embeddings_factory: Callable[[], Embeddings] | None = None,
    session_factory=SessionLocal,
) -> None:
    """Full load -> chunk -> embed -> store run, updating the tracking row.

    Runs in a FastAPI BackgroundTasks job. Any failure marks the document as
    failed with the error text; the temp file is always removed, even when
    ingestion fails mid-flight.
    """
    document_row_id = uuid_module.UUID(str(document_id))
    session = session_factory()
    try:
        try:
            documents = load_document(temp_path, filename)
            chunks = chunk_documents(documents)
            annotated = [
                _annotate(chunk, user_id=user_id, document_id=document_id, source=filename, index=position)
                for position, chunk in enumerate(chunks)
            ]
            if embeddings_factory is None:
                store = build_vectorstore()
            else:
                store = build_vectorstore(embeddings=embeddings_factory())
            store.add_documents(annotated)
        except Exception as error:  # noqa: BLE001 - every failure must be recorded
            row = session.get(DocumentRow, document_row_id)
            if row is not None:
                row.status = DocumentStatus.FAILED
                row.error = str(error)
                row.chunk_count = 0
                session.commit()
            logger.exception("Ingestion failed for document %s (%s): %s", document_id, filename, error)
            return

        row = session.get(DocumentRow, document_row_id)
        if row is not None:
            row.status = DocumentStatus.READY
            row.chunk_count = len(annotated)
            row.error = None
            session.commit()
        logger.info("Ingested %s as %s chunks", filename, len(annotated))
    finally:
        session.close()
        temp = Path(temp_path)
        try:
            if temp.exists():
                temp.unlink()
        except OSError:
            logger.warning("Could not delete temp file %s", temp_path)
