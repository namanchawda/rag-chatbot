"""Email delivery for account verification and password recovery."""

from __future__ import annotations

import base64
import logging
import threading
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formataddr

import httpx
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..config import get_settings, validate_email_settings
from ..db import SessionLocal
from ..models import EmailDailyUsage
from ..rag.status import SERVICE_EMAIL, record_failure, record_success

GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
EMAIL_LIMIT_MESSAGE = "Email delivery is temporarily unavailable. Please try again later."

_token_lock = threading.Lock()
_token_cache: dict = {"value": None, "expires_at": 0.0}


class EmailSendError(RuntimeError):
    """Client-safe delivery failure; never includes provider response details."""

    def __init__(self, http_status: int | None = None):
        self.http_status = http_status
        super().__init__(EMAIL_LIMIT_MESSAGE)


def _reserve_daily_send() -> bool:
    settings = get_settings()
    limit = settings.MAX_EMAILS_PER_DAY
    if limit <= 0:
        return False
    today = datetime.now(timezone.utc).date()
    with SessionLocal() as db:
        dialect = db.get_bind().dialect.name
        if dialect == "sqlite":
            statement = sqlite_insert(EmailDailyUsage)
        elif dialect == "postgresql":
            statement = postgres_insert(EmailDailyUsage)
        else:
            row = db.get(EmailDailyUsage, today)
            if row is None:
                row = EmailDailyUsage(day=today, sends=0)
                db.add(row)
                db.flush()
            if row.sends >= limit:
                return False
            db.execute(
                update(EmailDailyUsage)
                .where(EmailDailyUsage.day == today, EmailDailyUsage.sends < limit)
                .values(sends=EmailDailyUsage.sends + 1)
            )
            db.commit()
            return True

        result = db.execute(
            statement.values(day=today, sends=1)
            .on_conflict_do_update(
                index_elements=[EmailDailyUsage.day],
                set_={"sends": EmailDailyUsage.sends + 1},
                where=EmailDailyUsage.sends < limit,
            )
            .returning(EmailDailyUsage.sends)
        )
        reserved = result.scalar_one_or_none() is not None
        db.commit()
        return reserved


def _build_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(10.0))


logger = logging.getLogger(__name__)


def _google_error(response: httpx.Response) -> str:
    try:
        err = response.json().get("error")
    except ValueError:
        return "unknown"
    if isinstance(err, dict):
        return str(err.get("status") or err.get("code") or "unknown")
    return str(err or "unknown")


def _gmail_access_token(client: httpx.Client, settings, force: bool = False) -> str:
    with _token_lock:
        if not force and _token_cache["value"] and time.time() < _token_cache["expires_at"]:
            return _token_cache["value"]
        try:
            response = client.post(
                GMAIL_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": settings.GMAIL_CLIENT_ID,
                    "client_secret": settings.GMAIL_CLIENT_SECRET,
                    "refresh_token": settings.GMAIL_REFRESH_TOKEN,
                },
            )
        except httpx.HTTPError:
            logger.warning("Gmail token refresh failed: network error")
            raise EmailSendError() from None
        if not response.is_success:
            logger.warning(
                "Gmail token refresh failed (status=%s, error=%s)",
                response.status_code,
                _google_error(response),
            )
            raise EmailSendError(response.status_code)
        data = response.json()
        _token_cache["value"] = data["access_token"]
        _token_cache["expires_at"] = time.time() + int(data.get("expires_in", 3600)) - 60
        return _token_cache["value"]


def _gmail_send(to: str, subject: str, text: str, html: str) -> None:
    settings = get_settings()
    validate_email_settings(settings)

    clean = lambda v: v.replace("\r", " ").replace("\n", " ").strip()
    message = EmailMessage()
    message["From"] = formataddr((settings.EMAIL_FROM_NAME, settings.GMAIL_USER))
    message["To"] = clean(to)
    message["Subject"] = clean(subject)
    message.set_content(text)
    if html:
        message.add_alternative(html, subtype="html")
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")

    with _build_client() as client:
        token = _gmail_access_token(client, settings)
        for attempt in range(2):
            try:
                response = client.post(
                    GMAIL_SEND_URL,
                    headers={"Authorization": f"Bearer {token}"},
                    json={"raw": raw},
                )
            except httpx.HTTPError:
                if attempt == 0:
                    continue
                logger.warning("Gmail send failed: network error")
                raise EmailSendError() from None
            if response.is_success:
                return
            if response.status_code == 401 and attempt == 0:
                token = _gmail_access_token(client, settings, force=True)
                continue
            if response.status_code >= 500 and attempt == 0:
                continue
            logger.warning(
                "Gmail send failed (status=%s, error=%s)",
                response.status_code,
                _google_error(response),
            )
            raise EmailSendError(response.status_code)


def send_email(to: str, subject: str, text: str, html: str) -> None:
    """Send via the configured provider and record the outcome."""
    settings = get_settings()
    if not _reserve_daily_send():
        record_failure(SERVICE_EMAIL, 503, "Daily email delivery limit reached")
        raise EmailSendError(503)
    try:
        if settings.EMAIL_BACKEND == "console":
            print(f"\n--- Console email to {to} ---\nSubject: {subject}\n{text}\n")
        elif settings.EMAIL_BACKEND == "brevo":
            raise EmailSendError()
        elif settings.EMAIL_BACKEND == "gmail":
            _gmail_send(to, subject, text, html)
        else:
            raise EmailSendError()
    except EmailSendError as error:
        record_failure(SERVICE_EMAIL, error.http_status, "Email delivery failed")
        raise
    record_success(SERVICE_EMAIL)