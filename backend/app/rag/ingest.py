"""Document loading, chunking, and persistence into pgvector.

PDF-only ingestion with safety caps: page limit, per-document chunk limit,
and a per-user total chunk limit. Every cap is enforced before a single
Jina embedding token is spent. Embedding and insertion run in fixed-size
batches so memory stays flat, and a mid-flight batch failure rolls back
the vectors already inserted.
"""

import logging
import uuid as uuid_module
from collections.abc import Callable
from pathlib import Path

import fitz
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import func, select, text as sql_text
from sqlalchemy.engine import Engine

from ..config import get_settings
from ..db import SessionLocal, engine
from ..models import Document as DocumentRow
from ..models import DocumentStatus
from .embeddings import Embeddings, JinaEmbeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag-chatbot-chunks"
SUPPORTED_SUFFIXES = {".pdf"}

# About 64 chunks per embed+insert round trip keeps batch memory flat and
# limits the blast radius of a failed batch.
_INGEST_BATCH_SIZE = 64


class UnsupportedFileTypeError(ValueError):
    """Raised when a file suffix is not in the supported set."""


class PdfTooManyPagesError(ValueError):
    """Raised when a PDF exceeds MAX_PDF_PAGES."""


class ScannedPdfError(ValueError):
    """Raised when a PDF holds no extractable text."""


class EncryptedPdfError(ValueError):
    """Raised when a PDF requires a password."""


class CorruptPdfError(ValueError):
    """Raised when a PDF cannot be parsed at all."""


class ChunkLimitError(ValueError):
    """Raised when one document would exceed MAX_CHUNKS_PER_DOCUMENT."""


class UserChunkLimitError(ValueError):
    """Raised when a document would exceed MAX_TOTAL_CHUNKS_PER_USER."""


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


def _validate_pdf(resolved: str) -> None:
    """Fail fast on the PDF defects that would produce a broken ingest.

    Checks, in order: file is parseable, not password protected, page count
    under MAX_PDF_PAGES, and at least one page with extractable text. All
    checks run before any loader or embedding work.
    """
    settings = get_settings()
    try:
        doc = fitz.open(resolved)
    except Exception as error:  # noqa: BLE001 - any MuPDF parse failure is corruption
        raise CorruptPdfError(
            "This PDF is corrupt or unreadable. Try re-saving it and upload again."
        ) from error
    try:
        if doc.needs_pass:
            raise EncryptedPdfError(
                "This PDF is password protected. Remove the password and upload again."
            )
        if doc.page_count > settings.MAX_PDF_PAGES:
            raise PdfTooManyPagesError(
                f"This PDF has {doc.page_count} pages, above the "
                f"{settings.MAX_PDF_PAGES} page limit. Split it and upload the parts."
            )
        has_text = False
        for page in doc:
            if (page.get_text() or "").strip():
                has_text = True
                break
        if not has_text:
            raise ScannedPdfError("No readable text found. Scanned PDFs aren't supported.")
    finally:
        doc.close()


def load_document(temp_path: str | Path, filename: str) -> list[Document]:
    """Load a single temporary PDF; anything else is unsupported."""
    suffix = Path(filename).suffix.casefold()
    if suffix != ".pdf":
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Supported type: .pdf"
        )
    resolved = str(temp_path)
    _validate_pdf(resolved)
    return PyMuPDFLoader(resolved, mode="single").load()


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


def _embed_and_insert(store: PGVector, annotated: list[Document], document_id: str) -> None:
    """Embed and insert in fixed-size batches; roll back on failure.

    Each add_documents call embeds only its own batch, so peak memory stays
    flat. If a later batch fails, every vector already inserted for this
    document (ids "<document_id>:<i>") is deleted before the error propagates,
    leaving no orphans behind.
    """
    inserted = 0
    try:
        for start in range(0, len(annotated), _INGEST_BATCH_SIZE):
            batch = annotated[start : start + _INGEST_BATCH_SIZE]
            store.add_documents(batch)
            inserted += len(batch)
    except Exception:
        if inserted:
            try:
                store.delete(ids=[f"{document_id}:{index}" for index in range(inserted)])
            except Exception:  # noqa: BLE001 - rollback failure must not hide the original error
                logger.exception(
                    "Could not delete the %s vectors already inserted for document %s",
                    inserted,
                    document_id,
                )
        raise


def _ready_chunk_count_for_user(session, user_row_id: uuid_module.UUID) -> int:
    """Total chunks across the user's READY documents."""
    return int(
        session.scalar(
            select(func.coalesce(func.sum(DocumentRow.chunk_count), 0)).where(
                DocumentRow.user_id == user_row_id,
                DocumentRow.status == DocumentStatus.READY,
            )
        )
        or 0
    )


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

    The page cap, hollow-text check, per-document chunk cap, and per-user
    total cap all fail before the embedder is constructed, so no Jina tokens
    are spent on documents that can never be accepted.
    """
    document_row_id = uuid_module.UUID(str(document_id))
    session = session_factory()
    try:
        try:
            documents = load_document(temp_path, filename)
            chunks = chunk_documents(documents)
            settings = get_settings()

            if len(chunks) > settings.MAX_CHUNKS_PER_DOCUMENT:
                raise ChunkLimitError(
                    f"This document splits into {len(chunks)} chunks, above the "
                    f"{settings.MAX_CHUNKS_PER_DOCUMENT} chunk limit per document. "
                    "Try a shorter or smaller PDF."
                )

            ready_total = _ready_chunk_count_for_user(session, uuid_module.UUID(user_id))
            projected_total = ready_total + len(chunks)
            if projected_total > settings.MAX_TOTAL_CHUNKS_PER_USER:
                raise UserChunkLimitError(
                    f"Upload would push your documents to {projected_total} chunks, above the "
                    f"{settings.MAX_TOTAL_CHUNKS_PER_USER} chunk limit per user. "
                    "Delete an old document and try again."
                )

            annotated = [
                _annotate(chunk, user_id=user_id, document_id=document_id, source=filename, index=position)
                for position, chunk in enumerate(chunks)
            ]
            if embeddings_factory is None:
                store = build_vectorstore()
            else:
                store = build_vectorstore(embeddings=embeddings_factory())
            _embed_and_insert(store, annotated, document_id)
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
