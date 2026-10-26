"""Startup maintenance sweeps that never prevent the app from booting."""

import logging
from datetime import timedelta

from sqlalchemy import select

from .config import get_settings
from .db import SessionLocal
from .models import Document, DocumentStatus, _utcnow
from .rag.ingest import build_vectorstore, delete_document_vectors

logger = logging.getLogger(__name__)

STALE_UPLOAD_ERROR = "Upload was interrupted. Please delete it and upload again."


def fail_stale_uploads() -> int:
    """Mark long-stranded `processing` documents as failed and drop partial vectors.

    A crash between upload start and ingest completion can leave a document row
    that never leaves `processing`; the vector store may already hold chunks
    with ids prefixed "<document_id>:". Returns the number of rows failed.
    """
    cutoff = _utcnow() - timedelta(minutes=get_settings().STALE_PROCESSING_MINUTES)
    with SessionLocal() as db:
        stale = db.scalars(
            select(Document)
            .where(Document.status == DocumentStatus.PROCESSING)
            .where(Document.created_at < cutoff)
        ).all()
        if not stale:
            return 0
        store = None
        for row in stale:
            chunk_count = row.chunk_count
            row.status = DocumentStatus.FAILED
            row.error = STALE_UPLOAD_ERROR
            db.commit()
            if chunk_count > 0:
                if store is None:
                    store = build_vectorstore()
                delete_document_vectors(store, str(row.id), chunk_count)
    logger.warning("Failed %d stale uploads at startup", len(stale))
    return len(stale)
