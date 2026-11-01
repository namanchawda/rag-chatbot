"""Startup maintenance sweeps that never prevent the app from booting."""

import logging
from datetime import timedelta

from sqlalchemy import delete, select

from .config import get_settings
from .db import SessionLocal
from .models import AuthCode, Document, DocumentStatus, User, _utcnow
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


def cleanup_auth_records() -> tuple[int, int]:
    """Delete expired auth codes and long-abandoned unverified accounts."""
    now = _utcnow()
    settings = get_settings()
    unverified_cutoff = now - timedelta(hours=settings.UNVERIFIED_ACCOUNT_TTL_HOURS)
    code_cutoff = now - timedelta(days=1)
    with SessionLocal() as db:
        old_unverified_ids = select(User.id).where(
            User.email_verified.is_(False), User.created_at < unverified_cutoff
        )
        deleted_codes = db.execute(
            delete(AuthCode).where(
                AuthCode.created_at < code_cutoff,
                (AuthCode.used_at.is_not(None)) | (AuthCode.expires_at < code_cutoff),
            )
        ).rowcount or 0
        db.execute(delete(AuthCode).where(AuthCode.user_id.in_(old_unverified_ids)))
        deleted_users = db.execute(
            delete(User).where(User.id.in_(old_unverified_ids))
        ).rowcount or 0
        db.commit()
    return deleted_users, deleted_codes
