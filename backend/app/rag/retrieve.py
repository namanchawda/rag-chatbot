"""User-scoped retrieval from the pgvector store.

The similarity search runs inside PGVector with a metadata filter that is
part of the vector query itself, so chunks belonging to other users are
never fetched from the database, not merely filtered out afterwards.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..config import get_settings
from .ingest import build_vectorstore


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    document_id: str
    chunk_index: int


def user_filter(user_id: uuid.UUID | str) -> dict[str, str]:
    """Filter clause restricting a vector query to one user's chunks."""
    return {"user_id": str(user_id)}


def retrieve_chunks(
    question: str,
    user_id: uuid.UUID | str,
    store=None,
    *,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Search the store for the most relevant chunks owned by one user.

    PGVector calls embed_query (task retrieval.query) internally, and the
    user_id filter is handed to the same SQL query, so only the caller's
    own chunks leave the database.

    Args:
        question: The user's question, already trimmed and length-checked.
        user_id: The authenticated user's id; used verbatim in the filter.
        store: Optional PGVector instance (tests pass a fake). Defaults to
            the shared collection built by ingest.build_vectorstore().
        top_k: Optional override for RETRIEVAL_TOP_K.
    """
    if not isinstance(user_id, str):
        user_id = str(user_id)
    settings = get_settings()
    limit = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
    if store is None:
        store = build_vectorstore()
    documents = store.similarity_search(question, k=limit, filter=user_filter(user_id))
    chunks: list[RetrievedChunk] = []
    for document in documents:
        metadata = document.metadata or {}
        chunks.append(
            RetrievedChunk(
                text=document.page_content,
                source=str(metadata.get("source", "")),
                document_id=str(metadata.get("document_id", "")),
                chunk_index=int(metadata.get("chunk_index", 0)),
            )
        )
    return chunks
