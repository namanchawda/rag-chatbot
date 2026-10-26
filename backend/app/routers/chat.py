"""Chat endpoints: grounded answers over the caller's own documents.

user_id always comes from the authenticated token, never from the request
body. Retrieval is user-scoped through the vector-store filter. Every turn
lives in a chat session owned by the caller; the user turn is stored during
request validation, the assistant turn (including partial text after an
error, Stop, or disconnect) is saved in the background so a persistence
failure can never break an already-streamed answer.
"""

import asyncio
import json
import logging
import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from ..config import get_settings
from ..db import SessionLocal
from ..deps import get_current_user
from ..models import _utcnow, ChatMessage, ChatRole, ChatSession, Document, DocumentStatus, User
from ..rag.chat import (
    NO_DOCUMENTS_MESSAGE,
    NO_RESULTS_MESSAGE,
    build_llm,
    format_context,
    friendly_error,
    make_chain,
)
from ..rag.retrieve import RetrievedChunk, retrieve_reranked_chunks
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 200
_TITLE_CHARS = 60
_MESSAGE_WINDOW_HOURS = 24


def _session_not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Session not found",
    )


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


def _title_from_question(question: str) -> str:
    """Short whitespace-collapsed title for a brand-new session."""
    collapsed = " ".join(question.split())
    if len(collapsed) <= _TITLE_CHARS:
        return collapsed
    cut = collapsed[:_TITLE_CHARS]
    boundary_cut, separator, _ = cut.rpartition(" ")
    if separator and boundary_cut:
        return boundary_cut
    return cut


def _check_daily_message_cap(user_id: uuid.UUID) -> None:
    """Reject when the caller's own user turns in the rolling 24h window
    already reached MAX_MESSAGES_PER_DAY. Plain HTTP error, raised before any
    session or message row is created and before the stream starts.
    """
    limit = get_settings().MAX_MESSAGES_PER_DAY
    cutoff = _utcnow() - timedelta(hours=_MESSAGE_WINDOW_HOURS)
    statement = (
        select(func.count())
        .select_from(ChatMessage)
        .join(ChatSession, ChatMessage.session_id == ChatSession.id)
        .where(ChatSession.user_id == user_id)
        .where(ChatMessage.role == ChatRole.USER)
        .where(ChatMessage.created_at >= cutoff)
    )
    with SessionLocal() as session:
        recent = int(session.scalar(statement) or 0)
    if recent >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily message limit reached ({limit}). Try again tomorrow.",
        )


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


def _resolve_session_and_save_user(
    raw_session_id: str | None,
    user_id: uuid.UUID,
    question: str,
) -> tuple[uuid.UUID, str]:
    """Find or create the caller's session and store the user turn.

    Returns (session_id, title) for the stream's `session` event. A missing
    or foreign session id is a 404, indistinguishable from nonexistent.
    """
    try:
        session_id = uuid.UUID(str(raw_session_id)) if raw_session_id else None
    except (TypeError, ValueError, AttributeError):
        raise _session_not_found()

    with SessionLocal() as db:
        if session_id is None:
            row = ChatSession(user_id=user_id, title=_title_from_question(question))
        else:
            row = db.get(ChatSession, session_id)
            if row is None or row.user_id != user_id:
                raise _session_not_found()
        row.messages.append(ChatMessage(role=ChatRole.USER, content=question))
        row.updated_at = _utcnow()
        db.add(row)
        db.commit()
        return row.id, row.title


def _save_assistant_message(
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    content: str,
    sources: list[dict],
) -> None:
    with SessionLocal() as db:
        row = db.get(ChatSession, session_id)
        if row is None or row.user_id != user_id:
            return
        row.messages.append(
            ChatMessage(
                role=ChatRole.ASSISTANT,
                content=content,
                sources=sources or None,
            )
        )
        row.updated_at = _utcnow()
        db.commit()


def _persist_safely(
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    content: str,
    sources: list[dict],
) -> None:
    """Persistence must never turn an answer already given into an error."""
    try:
        _save_assistant_message(session_id, user_id, content, sources)
    except Exception:  # noqa: BLE001
        logger.exception("Saving assistant message failed; answer already delivered")


def _log_background_save(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.warning("Background chat message save failed: %s", error)


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
    _check_daily_message_cap(current_user.id)
    session_id, _title = _resolve_session_and_save_user(
        body.session_id, current_user.id, question
    )

    sources: list[dict] = []
    if not _user_has_ready_documents(current_user.id):
        answer = NO_DOCUMENTS_MESSAGE
    else:
        chunks = retrieve_reranked_chunks(question, current_user.id)
        if not chunks:
            answer = NO_RESULTS_MESSAGE
        else:
            sources = _sources(chunks)
            try:
                chain = make_chain(build_llm())
                answer = chain.invoke({"context": format_context(chunks), "question": question})
            except Exception as error:  # noqa: BLE001
                logger.exception("Chat answer failed for user %s", current_user.id)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=friendly_error(error),
                )

    _persist_safely(session_id, current_user.id, answer, sources)
    return ChatResponse(answer=answer, sources=sources, session_id=str(session_id))


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _chat_stream_events(
    question: str,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    session_title: str,
):
    """SSE body: session, sources, token..., done (or error).

    Accumulates streamed text and schedules ONE background save of the
    assistant turn when the generator ends, whatever way it ends (done,
    error, Stop button, client disconnect).
    """
    text_parts: list[str] = []
    sources: list[dict] = []
    saved = False

    def _schedule_persist() -> None:
        nonlocal saved
        content = "".join(text_parts)
        if saved or not content.strip():
            return
        saved = True
        task = asyncio.get_running_loop().create_task(
            run_in_threadpool(_persist_safely, session_id, user_id, content, sources)
        )
        task.add_done_callback(_log_background_save)

    try:
        yield _sse("session", {"session_id": str(session_id), "title": session_title})

        try:
            has_ready = await run_in_threadpool(_user_has_ready_documents, user_id)
        except Exception:  # noqa: BLE001
            logger.exception("Ready-document check failed for user %s", user_id)
            yield _sse("error", {"error": "Could not look up your documents. Please try again."})
            return

        if not has_ready:
            yield _sse("sources", {"sources": []})
            text_parts.append(NO_DOCUMENTS_MESSAGE)
            yield _sse("token", {"text": NO_DOCUMENTS_MESSAGE})
            yield _sse("done", {"status": "ok"})
            return

        try:
            # The blocking embed + SQL call happens in a worker thread so the
            # event loop stays free.
            chunks = await run_in_threadpool(retrieve_reranked_chunks, question, user_id)
        except Exception:  # noqa: BLE001
            logger.exception("Retrieval failed for user %s", user_id)
            yield _sse("error", {"error": "Could not search your documents. Please try again."})
            return

        sources = _sources(chunks)
        yield _sse("sources", {"sources": sources})

        if not chunks:
            text_parts.append(NO_RESULTS_MESSAGE)
            yield _sse("token", {"text": NO_RESULTS_MESSAGE})
            yield _sse("done", {"status": "ok"})
            return

        try:
            chain = make_chain(build_llm())
            async for text_delta in chain.astream(
                {"context": format_context(chunks), "question": question}
            ):
                if text_delta:
                    text_parts.append(text_delta)
                    yield _sse("token", {"text": text_delta})
        except Exception as error:  # noqa: BLE001
            logger.exception("Chat streaming failed for user %s", user_id)
            yield _sse("error", {"error": friendly_error(error)})
        yield _sse("done", {"status": "ok"})
    finally:
        _schedule_persist()


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Server-Sent Events stream: session, sources, token..., done (or error)."""

    question = _normalized_question(body.question)
    _check_daily_message_cap(current_user.id)
    # Resolve and persist the user turn up front, so a bad or foreign session
    # id is a clean 404 instead of an error event mid-stream.
    session_id, session_title = await run_in_threadpool(
        _resolve_session_and_save_user, body.session_id, current_user.id, question
    )

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _chat_stream_events(question, current_user.id, session_id, session_title),
        media_type="text/event-stream",
        headers=headers,
    )
