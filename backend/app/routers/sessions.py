"""Chat session and message endpoints.

Session ownership is checked on every request: a session id that belongs to
another user is indistinguishable from a missing one (404 either way).
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from ..db import SessionLocal
from ..deps import get_current_user
from ..models import ChatMessage, ChatSession, User
from ..schemas import MessageResponse, SessionCreate, SessionResponse, SessionUpdate

router = APIRouter(prefix="/sessions", tags=["sessions"])

_MAX_LIST_SESSIONS = 100
_MAX_LIST_MESSAGES = 500


def _session_not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Session not found",
    )


def _parse_session_id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(raw))
    except (TypeError, ValueError, AttributeError):
        raise _session_not_found()


def _owned_session(db, session_id: uuid.UUID, user_id: uuid.UUID) -> ChatSession:
    row = db.get(ChatSession, session_id)
    if row is None or row.user_id != user_id:
        raise _session_not_found()
    return row


@router.get("")
def list_sessions(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
) -> list[SessionResponse]:
    """The caller's sessions, most recently active first."""
    limit = max(1, min(limit, _MAX_LIST_SESSIONS))
    with SessionLocal() as db:
        statement = (
            select(ChatSession)
            .where(ChatSession.user_id == current_user.id)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
        )
        rows = db.scalars(statement).all()
    return [SessionResponse.model_validate(row) for row in rows]


@router.post("", status_code=status.HTTP_201_CREATED)
def create_session(
    body: SessionCreate,
    current_user: User = Depends(get_current_user),
) -> SessionResponse:
    row = ChatSession(user_id=current_user.id, title=body.title or "")
    with SessionLocal() as db:
        db.add(row)
        db.commit()
    return SessionResponse.model_validate(row)


@router.get("/{session_id}/messages")
def list_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
) -> list[MessageResponse]:
    """Stored turns of one session, oldest first (newest 500 kept)."""
    parsed = _parse_session_id(session_id)
    with SessionLocal() as db:
        _owned_session(db, parsed, current_user.id)
        statement = (
            select(ChatMessage)
            .where(ChatMessage.session_id == parsed)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(_MAX_LIST_MESSAGES)
        )
        rows = list(db.scalars(statement))
        rows.reverse()
    return [MessageResponse.model_validate(row) for row in rows]


@router.patch("/{session_id}")
def rename_session(
    session_id: str,
    body: SessionUpdate,
    current_user: User = Depends(get_current_user),
) -> SessionResponse:
    parsed = _parse_session_id(session_id)
    with SessionLocal() as db:
        row = _owned_session(db, parsed, current_user.id)
        row.title = body.title
        db.commit()
    return SessionResponse.model_validate(row)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
) -> None:
    parsed = _parse_session_id(session_id)
    with SessionLocal() as db:
        row = _owned_session(db, parsed, current_user.id)
        db.delete(row)
        db.commit()
