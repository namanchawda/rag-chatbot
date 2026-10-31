"""User-scoped retrieval from the pgvector store.

The similarity search runs inside PGVector with a metadata filter that is
part of the vector query itself, so chunks belonging to other users are
never fetched from the database, not merely filtered out afterwards.

An optional reranking stage scores the retrieved candidates with the Jina
rerank API and keeps only the best ones for the LLM.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from ..config import get_settings
from .ingest import build_vectorstore

logger = logging.getLogger(__name__)


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


def _log_scores(
    question: str,
    candidates: list[RetrievedChunk],
    scored: list[tuple[int, float]],
) -> None:
    """Log one INFO line per question: trunc. question + index:score pairs.

    Never logs chunk text or query-bearing secrets; scores are raw model
    outputs, not probabilities, so they are printed as given.
    """
    pairs = " ".join(f"{index}:{score:.4f}" for index, score in scored)
    logger.info(
        "Rerank scores question='%s' | %s",
        question[:60],
        pairs,
        extra={"candidate_count": len(candidates)},
    )


def resolve_rerank(requested: bool | None) -> bool:
    """Effective reranking decision for one request.

    RERANK_ENABLED=false is the master switch: it forces reranking off no
    matter what the caller asked for. Otherwise a request value wins, and
    None (or an omitted field) falls back to the server default, which is
    the master switch itself.
    """
    settings = get_settings()
    if not settings.RERANK_ENABLED:
        return False
    return settings.RERANK_ENABLED if requested is None else bool(requested)


def retrieve_scored_candidates(
    question: str,
    user_id: uuid.UUID | str,
    store=None,
    *,
    reranker=None,
    rerank: bool | None = None,
) -> tuple[list[tuple[RetrievedChunk, float | None]], bool]:
    """Every candidate paired with its rerank relevance score.

    Identical retrieval path as production (same store built by
    build_vectorstore, same in-query user_id filter), but no cutoff and no
    top_n slicing: every candidate keeps its score so RERANK_MIN_SCORE can
    be calibrated against observed values.

    The optional `rerank` argument is the per-request option; see
    resolve_rerank. Candidates are returned in vector order with None scores
    and a False flag when reranking is off or when the rerank call failed -
    the fallback production falls back to as well.
    """
    settings = get_settings()
    candidates = retrieve_chunks(
        question,
        user_id,
        store=store,
        top_k=settings.RETRIEVAL_CANDIDATES,
    )
    if not candidates:
        return [], False
    if not resolve_rerank(rerank):
        return [(chunk, None) for chunk in candidates], False

    try:
        if reranker is None:
            from .rerank import JinaReranker

            reranker = JinaReranker()
        # Score every candidate so min_score cutoffs and score logging see
        # the full ranking, not just an API-side truncated one.
        scored = reranker.rerank(question, [chunk.text for chunk in candidates])
    except Exception as error:  # noqa: BLE001 - failure must not break chat
        logger.warning(
            "Reranking failed (%s); falling back to vector order",
            type(error).__name__,
        )
        return [(chunk, None) for chunk in candidates], False

    if settings.RERANK_LOG_SCORES:
        _log_scores(question, candidates, scored)

    # Map by index, never by position: the API returns results in relevance
    # order, which is unrelated to the candidate order we sent.
    ranked = [
        (candidates[index], score)
        for index, score in sorted(scored, key=lambda pair: pair[1], reverse=True)
        if 0 <= index < len(candidates)
    ]
    return ranked, True


def _final_chunks(
    scored: list[tuple[RetrievedChunk, float | None]],
    reranked: bool,
) -> list[RetrievedChunk]:
    """Cutoff and RERANK_TOP_N slicing, only when real scores exist."""
    if not scored:
        return []
    settings = get_settings()
    top_n = settings.RERANK_TOP_N
    if reranked:
        cutoff = settings.RERANK_MIN_SCORE
        scored = [
            (chunk, score)
            for chunk, score in scored
            if score is not None and (cutoff is None or score >= cutoff)
        ]
    return [chunk for chunk, _ in scored[:top_n]]


def retrieve_reranked_chunks(
    question: str,
    user_id: uuid.UUID | str,
    store=None,
    *,
    reranker=None,
    rerank: bool | None = None,
) -> list[RetrievedChunk]:
    """Fetch candidates, optionally rerank them, and return the final set.

    Built on retrieve_scored_candidates; the cutoff and RERANK_TOP_N slicing
    only run when real scores exist (rerank on and successful).

    An empty return value means "nothing relevant found" and callers must
    skip the LLM entirely.
    """
    chunks, _ = retrieve_reranked_result(
        question,
        user_id,
        store=store,
        reranker=reranker,
        rerank=rerank,
    )
    return chunks


def retrieve_reranked_result(
    question: str,
    user_id: uuid.UUID | str,
    store=None,
    *,
    reranker=None,
    rerank: bool | None = None,
) -> tuple[list[RetrievedChunk], bool]:
    """Final chunk set plus whether reranking really ran.

    reranked is True only when rerank actually executed and produced scores;
    it is False when the option is off, overridden by the master switch, or
    when a rerank failure fell back to vector order.
    """
    scored, reranked = retrieve_scored_candidates(
        question,
        user_id,
        store=store,
        reranker=reranker,
        rerank=rerank,
    )
    return _final_chunks(scored, reranked), reranked
