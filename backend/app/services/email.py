"""Email delivery for account verification and password recovery."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..config import get_settings, validate_email_settings
from ..db import SessionLocal
from ..models import EmailDailyUsage
from ..rag.status import SERVICE_EMAIL, record_failure, record_success

BREVO_URL = "https://api.brevo.com/v3/smtp/email"
EMAIL_LIMIT_MESSAGE = "Email delivery is temporarily unavailable. Please try again later."


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


def _brevo_send(to: str, subject: str, text: str, html: str) -> None:
    settings = get_settings()
    validate_email_settings(settings)
    payload = {
        "sender": {
            "name": settings.EMAIL_FROM_NAME,
            "email": settings.EMAIL_FROM_ADDRESS,
        },
        "to": [{"email": to}],
        "subject": subject,
        "htmlContent": html,
        "textContent": text,
    }
    with _build_client() as client:
        for attempt in range(2):
            try:
                response = client.post(
                    BREVO_URL,
                    headers={"api-key": settings.BREVO_API_KEY},
                    json=payload,
                )
            except httpx.HTTPError:
                if attempt == 0:
                    continue
                raise EmailSendError() from None
            if response.is_success:
                return
            if response.status_code >= 500 and attempt == 0:
                continue
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
            _brevo_send(to, subject, text, html)
        else:
            raise EmailSendError()
    except EmailSendError as error:
        record_failure(SERVICE_EMAIL, error.http_status, "Email delivery failed")
        raise
    record_success(SERVICE_EMAIL)
