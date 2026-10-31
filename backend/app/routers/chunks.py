"""Read-only chunks browser: everything shown is scoped by the token's
user_id inside the SQL query itself, mirrors the other routers' style.

A foreign document_id or chunk id is 404, never 403 - ownership missing and
existence missing are indistinguishable.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import select

from ..db import SessionLocal
from ..deps import get_current_user
from ..models import Document, DocumentStatus as _DocumentStatus, User
from ..rag import chunks as chunks_store

router = APIRouter(prefix="/chunks", tags=["chunks"])


class ChunkItem(BaseModel):
    id: str
    document_id: str
    source: str
    chunk_index: int
    text: str
    char_count: int
    token_estimate: int


class ChunkListResponse(BaseModel):
    items: list[ChunkItem]
    total: int
    limit: int
    offset: int


class ChunkStatsResponse(BaseModel):
    documents: int
    chunks: int
    characters: int


def _str_of_current_user(user: User) -> str:
    # user_id ALWAYS comes from the token, never from any request input.
    return str(user.id)


def _owned_document_id(document_id: str, user_id: str) -> str | None:
    """The caller's document id as stored in chunk metadata; None -> 404."""
    try:
        parsed = uuid.UUID(document_id)
    except (TypeError, ValueError, AttributeError):
        return None
    with SessionLocal() as db:
        found = db.scalar(
            select(Document.id).where(
                Document.id == parsed, Document.user_id == uuid.UUID(user_id)
            )
        )
    if found is None:
        return None
    return str(found)


@router.get("", response_model=ChunkListResponse)
async def list_chunks(
    document_id: str | None = None,
    q: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=20, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
) -> ChunkListResponse:
    """Paginated view of the caller's chunks, filename then chunk_index order.

    limit is capped at 50 (default 20); q is a case-insensitive substring
    match on chunk text with %, _ and quotes matching literally.
    """
    user_id = _str_of_current_user(current_user)
    scoped_document_id = None
    if document_id is not None:
        scoped_document_id = await run_in_threadpool(
            _owned_document_id, document_id, user_id
        )
        if scoped_document_id is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )
    items, total = await run_in_threadpool(
        chunks_store.list_chunks,
        user_id,
        document_id=scoped_document_id,
        q=q,
        limit=limit,
        offset=offset,
    )
    return ChunkListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/stats", response_model=ChunkStatsResponse)
async def chunk_stats(
    current_user: User = Depends(get_current_user),
) -> ChunkStatsResponse:
    """{documents, chunks, characters} across the caller's READY documents."""
    user_id = _str_of_current_user(current_user)

    def ready_ids() -> list[str]:
        with SessionLocal() as db:
            rows = db.scalars(
                select(Document.id).where(
                    Document.user_id == uuid.UUID(user_id),
                    Document.status == _DocumentStatus.READY,
                )
            ).all()
        return [str(row) for row in rows]

    ids = await run_in_threadpool(ready_ids)
    stats = await run_in_threadpool(chunks_store.chunk_stats, user_id, ids)
    return ChunkStatsResponse(**stats)


@router.get("/{chunk_id}", response_model=ChunkItem)
async def get_chunk(
    chunk_id: str,
    current_user: User = Depends(get_current_user),
) -> ChunkItem:
    """One full chunk, owner only; a foreign id is a plain 404."""
    user_id = _str_of_current_user(current_user)
    item = await run_in_threadpool(chunks_store.get_chunk, user_id, chunk_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found"
        )
    return ChunkItem(**item)
