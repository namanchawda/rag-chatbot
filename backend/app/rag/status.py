"""Best-effort service usage and status tracking for Jina and Groq.

Every write goes to the small service_usage / service_status tables and is
wrapped in try/except: a recording failure is logged (type name only, no
secrets) and never raises or slows the request it belongs to. Usage counters
are bumped with atomic UPDATE statements; a lost INSERT race is retried once
as an UPDATE. Stored messages are sanitized: anything resembling an API key
is redacted, whitespace is collapsed, and the text is truncated. Request
bodies and document text are never stored.
"""

import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from ..config import get_settings
from ..db import SessionLocal
from ..models import ChatMessage, ChatRole, ServiceStatus, ServiceUsage

logger = logging.getLogger(__name__)

SERVICE_JINA = "jina"
SERVICE_GROQ = "groq"

STATE_OK = "ok"
STATE_INVALID_KEY = "invalid_key"
STATE_EXHAUSTED = "exhausted"
STATE_RATE_LIMITED = "rate_limited"
STATE_UNAVAILABLE = "unavailable"
STATE_ERROR = "error"
STATE_UNKNOWN = "unknown"

# States in which Jina cannot serve requests at all and callers must degrade.
BLOCKING_STATES = frozenset({STATE_EXHAUSTED, STATE_INVALID_KEY})

MESSAGE_MAX_CHARS = 200

_QUOTA_BODY_RE = re.compile(r"balance|insufficient|quota|exceeded", re.IGNORECASE)

_SECRET_PATTERNS = (
    # Whole "Bearer <token>" runs first so the following prefix patterns
    # cannot split it up; then bare jina_/gsk_ tokens.
    re.compile(r"bearer\s+[a-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"jina_[a-z0-9_\-]+", re.IGNORECASE),
    re.compile(r"gsk_[a-z0-9_\-]+", re.IGNORECASE),
)
_SECRET_REPLACEMENT = "[REDACTED]"

SERVICE_UNAVAILABLE_MESSAGE = (
    "The document processing service has reached its usage limit. Please try again later."
)

_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
_STATUS_CHECK_TEXT = "This is a tiny status probe."


class ServiceUnavailableError(RuntimeError):
    """Jina is exhausted (or its key is invalid); callers must degrade politely.

    str(error) is a client-safe message, so background ingest can store it
    directly as the document error text.
    """

    def __init__(self, message: str = SERVICE_UNAVAILABLE_MESSAGE):
        super().__init__(message)


def sanitize_message(text: str | None) -> str | None:
    """Redact key-like strings, collapse whitespace, truncate to 200 chars."""
    if text is None:
        return None
    cleaned = str(text)
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(_SECRET_REPLACEMENT, cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > MESSAGE_MAX_CHARS:
        cleaned = cleaned[: MESSAGE_MAX_CHARS - 3].rstrip() + "..."
    return cleaned


def usage_tokens(payload: Any) -> int:
    """Extract usage.total_tokens from a Jina response payload, else 0."""
    if not isinstance(payload, dict):
        return 0
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, (int, float)) and total >= 0:
        return int(total)
    return 0


def classify_failure(
    service: str,
    http_status: int | None,
    body_text: str | None,
    *,
    retry_after: str | None = None,
) -> tuple[str, str]:
    """Map one failed call to (state, sanitized message).

    Jina: 401/403 invalid_key; 402 exhausted; 429 rate_limited; any other 4xx
    whose body mentions balance/insufficient/quota/exceeded exhausted; 5xx or
    network errors unavailable; everything else error.

    Groq: 401 invalid_key; 429 rate_limited (Retry-After in the message when
    present); 5xx or network errors unavailable; everything else error.
    """
    if service not in (SERVICE_JINA, SERVICE_GROQ):
        raise ValueError(f"Unknown service: {service}")
    body = body_text or ""
    if service == SERVICE_JINA:
        if http_status in (401, 403):
            state, message = STATE_INVALID_KEY, body
        elif http_status == 402:
            state, message = STATE_EXHAUSTED, body
        elif http_status == 429:
            state, message = STATE_RATE_LIMITED, body
        elif (
            http_status is not None
            and 400 <= http_status < 500
            and _QUOTA_BODY_RE.search(body)
        ):
            state, message = STATE_EXHAUSTED, body
        elif http_status is None or (http_status >= 500):
            state, message = STATE_UNAVAILABLE, body
        else:
            state, message = STATE_ERROR, body
    else:  # Groq
        if http_status == 401:
            state, message = STATE_INVALID_KEY, body
        elif http_status == 429:
            state = STATE_RATE_LIMITED
            message = f"Rate limited; retry after {retry_after} seconds. {body}" if retry_after else body
        elif http_status is None or (http_status >= 500):
            state, message = STATE_UNAVAILABLE, body
        else:
            state, message = STATE_ERROR, body
    return state, sanitize_message(message) or ""


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _utcnow():
    return datetime.now(timezone.utc)


def _bump_usage(db, service: str, day: date, *, tokens: int = 0, errors: int = 0) -> None:
    """Atomic increment; on the unique-constraint insert race retry once as an UPDATE."""
    increment = (
        update(ServiceUsage)
        .where(ServiceUsage.service == service, ServiceUsage.day == day)
        .values(
            requests=ServiceUsage.requests + 1,
            tokens=ServiceUsage.tokens + tokens,
            errors=ServiceUsage.errors + errors,
        )
    )
    result = db.execute(increment)
    if result.rowcount:
        db.commit()
        return
    db.add(ServiceUsage(service=service, day=day, requests=1, tokens=tokens, errors=errors))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        db.execute(increment)
        db.commit()


def _record_outcome(
    service: str,
    *,
    tokens: int,
    errors: int,
    failure: tuple[str, int | None, str] | None,
) -> None:
    """Write the usage bump and the status row. Raises on DB failure."""
    now = _utcnow()
    with SessionLocal() as db:
        _bump_usage(db, service, _today(), tokens=tokens, errors=errors)
        row = db.get(ServiceStatus, service)
        if row is None:
            row = ServiceStatus(service=service, state=STATE_UNKNOWN)
            db.add(row)
            db.flush()
        if failure is not None:
            state, http_status, message = failure
            row.state = state
            row.http_status = http_status
            row.message = message
            row.last_error_at = now
        else:
            row.state = STATE_OK
            row.http_status = None
            row.message = None
            row.last_success_at = now
        row.updated_at = now
        db.commit()


def record_success(service: str, tokens: int = 0) -> None:
    """Record one successful call. Best-effort, never raises."""
    try:
        _record_outcome(service, tokens=tokens, errors=0, failure=None)
    except Exception as error:  # noqa: BLE001 - recording must never break the request
        logger.warning(
            "Could not record %s success (%s)", service, type(error).__name__
        )


def record_failure(
    service: str,
    http_status: int | None,
    body_text: str | None,
    *,
    retry_after: str | None = None,
) -> None:
    """Classify and record one failed call. Best-effort, never raises."""
    try:
        state, message = classify_failure(
            service, http_status, body_text, retry_after=retry_after
        )
        _record_outcome(
            service,
            tokens=0,
            errors=1,
            failure=(state, http_status, message),
        )
    except Exception as error:  # noqa: BLE001 - recording must never break the request
        logger.warning(
            "Could not record %s failure (%s)", service, type(error).__name__
        )


def jina_block_state() -> str | None:
    """'exhausted' or 'invalid_key' when Jina is currently unusable, else None."""
    try:
        with SessionLocal() as db:
            row = db.get(ServiceStatus, SERVICE_JINA)
    except Exception as error:  # noqa: BLE001 - without data the service is assumed up
        logger.warning("Could not read %s status (%s)", SERVICE_JINA, type(error).__name__)
        return None
    if row is not None and row.state in BLOCKING_STATES:
        return row.state
    return None


def reset_usage(service: str = SERVICE_JINA) -> None:
    """Set usage_reset_at to now, e.g. after replacing the Jina key.

    Day rows only know the date, not the time, so the current day's token
    bucket is zeroed too - otherwise tokens spent before the swap would
    forever count against the new key. requests/errors stay as call counts.
    """
    now = _utcnow()
    with SessionLocal() as db:
        row = db.get(ServiceStatus, service)
        if row is None:
            row = ServiceStatus(service=service, state=STATE_UNKNOWN)
            db.add(row)
        row.usage_reset_at = now
        row.updated_at = now
        db.execute(
            update(ServiceUsage)
            .where(ServiceUsage.service == service, ServiceUsage.day == now.date())
            .values(tokens=0)
        )
        db.commit()


def build_check_client() -> httpx.Client:
    """One fresh client per check; tests inject a MockTransport-backed one."""
    return httpx.Client(timeout=httpx.Timeout(15.0))


def run_jina_status_check(client: httpx.Client | None = None) -> None:
    """ONE minimal Jina embedding request of a short fixed text, recorded.

    No retries: whatever the single call reports is the recorded outcome.
    Network errors are recorded with http_status None.
    """
    settings = get_settings()
    own_client = client is None
    client = client or build_check_client()
    try:
        try:
            response = client.post(
                _EMBEDDINGS_URL,
                headers={
                    "Authorization": f"Bearer {settings.JINA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.JINA_EMBEDDING_MODEL,
                    "task": "retrieval.query",
                    "normalized": True,
                    "dimensions": settings.EMBEDDING_DIMENSION,
                    "input": [{"text": _STATUS_CHECK_TEXT}],
                },
            )
        except httpx.HTTPError as error:
            record_failure(SERVICE_JINA, None, f"Jina status check failed: {error}")
            return
        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            record_success(SERVICE_JINA, usage_tokens(payload))
        else:
            record_failure(SERVICE_JINA, response.status_code, response.text)
    finally:
        if own_client:
            client.close()


def status_payload() -> dict:
    """The admin payload for both services plus checked_at. Never has keys."""
    now = _utcnow()
    with SessionLocal() as db:
        jina_row = db.get(ServiceStatus, SERVICE_JINA)
        day = now.date()
        today_usage = db.scalar(
            select(ServiceUsage).where(ServiceUsage.service == SERVICE_JINA, ServiceUsage.day == day)
        )

        reset_at = jina_row.usage_reset_at if jina_row is not None else None
        since_query = select(func.coalesce(func.sum(ServiceUsage.tokens), 0)).where(
            ServiceUsage.service == SERVICE_JINA
        )
        if reset_at is not None:
            since_query = since_query.where(ServiceUsage.day >= reset_at.date())
        tokens_since_reset = int(db.scalar(since_query) or 0)

        answers_last_24h = int(
            db.scalar(
                select(func.count())
                .select_from(ChatMessage)
                .where(ChatMessage.role == ChatRole.ASSISTANT)
                .where(ChatMessage.created_at >= now - timedelta(hours=24))
            )
            or 0
        )

        groq_row = db.get(ServiceStatus, SERVICE_GROQ)

    settings = get_settings()
    budget = settings.JINA_TOKEN_BUDGET
    state = jina_row.state if jina_row is not None else STATE_UNKNOWN
    if state == STATE_EXHAUSTED:
        remaining = 0
    else:
        remaining = max(0, budget - tokens_since_reset)
    percent_remaining = int(remaining * 100 / budget) if budget > 0 else 0

    return {
        "jina": {
            "state": state,
            "http_status": jina_row.http_status if jina_row is not None else None,
            "message": jina_row.message if jina_row is not None else None,
            "last_success_at": jina_row.last_success_at if jina_row is not None else None,
            "last_error_at": jina_row.last_error_at if jina_row is not None else None,
            "tokens_used_since_reset": tokens_since_reset,
            "tokens_used_today": int(today_usage.tokens) if today_usage is not None else 0,
            "requests_today": int(today_usage.requests) if today_usage is not None else 0,
            "errors_today": int(today_usage.errors) if today_usage is not None else 0,
            "tokens_budget": budget,
            "tokens_estimated_remaining": remaining,
            "percent_remaining": percent_remaining,
            "low": percent_remaining < settings.JINA_LOW_BALANCE_PERCENT,
        },
        "groq": {
            "state": groq_row.state if groq_row is not None else STATE_UNKNOWN,
            "message": groq_row.message if groq_row is not None else None,
            "last_success_at": groq_row.last_success_at if groq_row is not None else None,
            "last_error_at": groq_row.last_error_at if groq_row is not None else None,
            "answers_last_24h": answers_last_24h,
        },
        "checked_at": now,
    }
