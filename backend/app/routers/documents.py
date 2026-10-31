import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import get_current_user
from ..models import Document, DocumentStatus, User
from ..rag.ingest import SUPPORTED_SUFFIXES, ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])


def _document_response(document: Document) -> dict:
    return {
        "id": str(document.id),
        "filename": document.filename,
        "status": document.status.value,
        "chunk_count": document.chunk_count,
        "error": document.error,
        "created_at": document.created_at,
    }


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    limit = get_settings().MAX_DOCUMENTS_PER_USER
    owned = db.scalar(select(func.count()).select_from(Document).where(Document.user_id == current_user.id)) or 0
    if owned >= limit:
        # Quota first: reject before reading a byte of the file or saving it.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document limit reached ({limit}). Delete one to upload another.",
        )

    filename = file.filename or ""
    if not filename.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided")

    suffix = Path(filename).suffix.casefold()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Supported type: .pdf",
        )

    max_bytes = get_settings().MAX_UPLOAD_MB * 1024 * 1024
    temp_dir = tempfile.gettempdir()
    descriptor, temp_path = tempfile.mkstemp(suffix=suffix, dir=temp_dir)
    size_bytes = 0
    try:
        with os.fdopen(descriptor, "wb") as buffer:
            while True:
                piece = file.file.read(1024 * 1024)
                if not piece:
                    break
                size_bytes += len(piece)
                if size_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the {get_settings().MAX_UPLOAD_MB} MB upload limit",
                    )
                buffer.write(piece)
        # Content sniffing: a renamed MIME type is not proof of a real PDF.
        with open(temp_path, "rb") as probe:
            magic = probe.read(5)
        if magic != b"%PDF-":
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="This file is not a valid PDF. Only real PDF files are accepted.",
            )
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise

    document = Document(
        id=uuid.uuid4(),
        user_id=current_user.id,
        filename=filename,
        status=DocumentStatus.PROCESSING,
        chunk_count=0,
        error=None,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    background_tasks.add_task(
        ingest_document,
        str(document.id),
        temp_path,
        filename,
        str(current_user.id),
    )

    return _document_response(document)


@router.get("")
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    rows = db.scalars(
        select(Document).where(Document.user_id == current_user.id).order_by(Document.created_at.desc())
    ).all()
    return [_document_response(document) for document in rows]


@router.get("/{document_id}")
def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    document = _get_owned_document(document_id, current_user, db)
    return _document_response(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    document = _get_owned_document(document_id, current_user, db)
    chunk_count = document.chunk_count

    if chunk_count > 0:
        # Vectors carry deterministic ids "<document_id>:<i>", so they can be
        # removed without querying the store by metadata.
        from ..rag.ingest import build_vectorstore, delete_document_vectors

        delete_document_vectors(build_vectorstore(), document_id, chunk_count)

    db.delete(document)
    db.commit()


def _get_owned_document(document_id: str, current_user: User, db: Session) -> Document:
    try:
        parsed_id = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    document = db.get(Document, parsed_id)
    if document is None or document.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document
