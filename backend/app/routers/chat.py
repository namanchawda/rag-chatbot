"""Chat endpoints: grounded answers over the caller's own documents.

user_id always comes from the authenticated token, never from the request
body. Retrieval is user-scoped through the vector-store filter.
"""

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from ..config import get_settings
from ..db import SessionLocal
from ..deps import get_current_user
from ..models import Document, DocumentStatus, User
from ..rag.chat import (
    NO_DOCUMENTS_MESSAGE,
    NO_RESULTS_MESSAGE,
    build_llm,
    format_context,
    friendly_error,
    make_chain,
)
from ..rag.retrieve import RetrievedChunk, retrieve_chunks
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 200


def _normalized_question(raw_question: str) -> str:
    question = (raw_question or "").strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty",
        )
    character_limit = get_settings().MAX_QUESTION_CHARS
    if len(question) > character_limit:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Question is too long; it must be at most {character_limit} characters",
        )
    return question


def _user_has_ready_documents(user_id: uuid.UUID) -> bool:
    """True when the user has at least one document with ingested chunks.

    Opens its own session so it is safe to run in a worker thread.
    """
    statement = (
        select(Document.id)
        .where(Document.user_id == user_id)
        .where(Document.status == DocumentStatus.READY)
        .where(Document.chunk_count > 0)
        .limit(1)
    )
    with SessionLocal() as session:
        return session.scalar(statement) is not None


def _sources(chunks: list[RetrievedChunk]) -> list[dict]:
    """De-duplicated source list, one entry per document chunk."""
    seen: set[tuple[str, int]] = set()
    sources: list[dict] = []
    for chunk in chunks:
        key = (chunk.document_id, chunk.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        ellipsis = "..." if len(chunk.text) > _SNIPPET_CHARS else ""
        sources.append(
            {
                "document_id": chunk.document_id,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "snippet": chunk.text[:_SNIPPET_CHARS] + ellipsis,
            }
        )
    return sources


@router.post("")
def ask_chat(body: ChatRequest, current_user: User = Depends(get_current_user)) -> ChatResponse:
    """Answer one question from the caller's documents (for testing in /docs)."""
    question = _normalized_question(body.question)

    if not _user_has_ready_documents(current_user.id):
        return ChatResponse(answer=NO_DOCUMENTS_MESSAGE, sources=[])

    chunks = retrieve_chunks(question, current_user.id)
    if not chunks:
        return ChatResponse(answer=NO_RESULTS_MESSAGE, sources=[])

    try:
        chain = make_chain(build_llm())
        answer = chain.invoke({"context": format_context(chunks), "question": question})
    except Exception as error:  # noqa: BLE001 - mapped to a friendly message below
        logger.exception("Chat answer failed for user %s", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=friendly_error(error),
        )
    return ChatResponse(answer=answer, sources=_sources(chunks))


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Server-Sent Events stream: sources, token..., done (or error)."""

    question = _normalized_question(body.question)
    user_id = current_user.id

    async def event_stream():
        try:
            has_ready = await run_in_threadpool(_user_has_ready_documents, user_id)
        except Exception:  # noqa: BLE001
            logger.exception("Ready-document check failed for user %s", user_id)
            yield _sse("error", {"error": "Could not look up your documents. Please try again."})
            return

        if not has_ready:
            yield _sse("sources", {"sources": []})
            yield _sse("token", {"text": NO_DOCUMENTS_MESSAGE})
            yield _sse("done", {"status": "ok"})
            return

        try:
            # The blocking embed + SQL call happens in a worker thread so the
            # event loop stays free.
            chunks = await run_in_threadpool(retrieve_chunks, question, user_id)
        except Exception:  # noqa: BLE001
            logger.exception("Retrieval failed for user %s", user_id)
            yield _sse("error", {"error": "Could not search your documents. Please try again."})
            return

        yield _sse("sources", {"sources": _sources(chunks)})

        if not chunks:
            yield _sse("token", {"text": NO_RESULTS_MESSAGE})
            yield _sse("done", {"status": "ok"})
            return

        try:
            chain = make_chain(build_llm())
            async for text_delta in chain.astream(
                {"context": format_context(chunks), "question": question}
            ):
                if text_delta:
                    yield _sse("token", {"text": text_delta})
        except Exception as error:  # noqa: BLE001
            logger.exception("Chat streaming failed for user %s", user_id)
            yield _sse("error", {"error": friendly_error(error)})
        yield _sse("done", {"status": "ok"})

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
