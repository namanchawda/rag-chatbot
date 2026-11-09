import base64
import json
import re
from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app import config as config_module
from app import rate_limit
from app.config import get_settings
from app.db import SessionLocal
from app.main import create_app
from app.maintenance import cleanup_auth_records
from app.models import AuthCode, EmailDailyUsage, User, _utcnow
from app.routers import auth as auth_router
from app.security import hash_password
from app.services import email as email_service
from app.services.email import EmailSendError


@pytest.fixture()
def fake_email(monkeypatch):
    sent = []

    def send(to, subject, text, html):
        sent.append({"to": to, "subject": subject, "text": text, "html": html})

    monkeypatch.setattr(auth_router, "send_email", send)
    return sent


@pytest.fixture(autouse=True)
def allow_auth_test_requests(monkeypatch):
    original = get_settings()
    monkeypatch.setattr(
        auth_router,
        "get_settings",
        lambda: original.model_copy(update={"AUTH_RATE_LIMIT_PER_MINUTE": 100}),
    )
    rate_limit.reset()
    yield
    rate_limit.reset()


def _client():
    return TestClient(create_app())


def _code(message):
    match = re.search(r"\b(\d{6})\b", message["text"])
    assert match is not None
    return match.group(1)


def _signup(client, email="new@example.com", password="long-enough-password"):
    return client.post("/auth/signup", json={"email": email, "password": password})


def _verify(client, email, code):
    return client.post("/auth/verify-email", json={"email": email, "code": code})


def _age_latest_code(db_session, user_id, seconds=61):
    row = db_session.scalar(
        select(AuthCode)
        .where(AuthCode.user_id == user_id)
        .order_by(AuthCode.created_at.desc())
        .limit(1)
    )
    assert row is not None
    row.created_at = _utcnow() - timedelta(seconds=seconds)
    db_session.commit()


def test_signup_verify_login_and_unverified_login_behavior(
    db_session, fake_email, monkeypatch
):
    with _client() as client:
        signup = _signup(client)
        assert signup.status_code == 201
        assert signup.json() == {
            "status": "verification_required",
            "email": "new@example.com",
        }
        assert "access_token" not in signup.json()
        assert "user" not in signup.json()
        code = _code(fake_email[-1])
        duplicate_unverified = _signup(client)
        assert duplicate_unverified.status_code == 201
        assert duplicate_unverified.json() == signup.json()
        assert len(fake_email) == 1

        with SessionLocal() as db:
            user = db.scalar(select(User).where(User.email == "new@example.com"))
            assert user is not None and user.email_verified is False
            stored = db.scalar(select(AuthCode).where(AuthCode.user_id == user.id))
            assert stored is not None and stored.code_hash != code
            assert code not in stored.code_hash

        wrong_password = client.post(
            "/auth/login",
            json={"email": "new@example.com", "password": "wrong-password"},
        )
        assert wrong_password.status_code == 401
        assert len(fake_email) == 1

        _age_latest_code(db_session, user.id)
        monkeypatch.setattr(
            auth_router,
            "get_settings",
            lambda: get_settings().model_copy(
                update={
                    "AUTH_RATE_LIMIT_PER_MINUTE": 100,
                    "EMAIL_RESEND_COOLDOWN_SECONDS": 0,
                }
            ),
        )
        correct_password = client.post(
            "/auth/login",
            json={"email": "new@example.com", "password": "long-enough-password"},
        )
        assert correct_password.status_code == 403
        assert correct_password.json() == {
            "detail": "Please verify your email to continue.",
            "code": "email_not_verified",
            "email": "new@example.com",
        }
        assert code not in correct_password.text
        assert len(fake_email) == 2
        assert _verify(client, "new@example.com", code).status_code == 400
        verification = _verify(client, "new@example.com", _code(fake_email[-1]))
        assert verification.status_code == 200
        assert _code(fake_email[-1]) not in verification.text
        assert "access_token" in verification.json()
        assert verification.json()["user"]["email"] == "new@example.com"
        login = client.post(
            "/auth/login",
            json={"email": "new@example.com", "password": "long-enough-password"},
        )
        assert login.status_code == 200
        assert login.json()["token_type"] == "bearer"
        assert _signup(client).status_code == 409


def test_wrong_expired_reused_and_over_attempt_codes(db_session, fake_email, monkeypatch):
    monkeypatch.setattr(
        auth_router,
        "get_settings",
        lambda: get_settings().model_copy(
            update={"AUTH_RATE_LIMIT_PER_MINUTE": 100, "CODE_MAX_ATTEMPTS": 2}
        ),
    )
    with _client() as client:
        _signup(client)
        first_code = _code(fake_email[-1])
        wrong = _verify(client, "new@example.com", "123456" if first_code != "123456" else "654321")
        assert wrong.status_code == 400
        assert wrong.json()["detail"] == "Invalid or expired code"
        wrong_again = _verify(
            client,
            "new@example.com",
            "123456" if first_code != "123456" else "654321",
        )
        assert wrong_again.status_code == 400
        assert _verify(client, "new@example.com", first_code).status_code == 400
        with SessionLocal() as db:
            row = db.scalar(select(AuthCode))
            assert row.attempts == 2

    with _client() as client:
        _signup(client, "expired@example.com")
        expired_code = _code(fake_email[-1])
        with SessionLocal() as db:
            user = db.scalar(select(User).where(User.email == "expired@example.com"))
            row = db.scalar(
                select(AuthCode).where(
                    AuthCode.user_id == user.id,
                    AuthCode.purpose == "verify",
                )
            )
            row.expires_at = _utcnow() - timedelta(seconds=1)
            code_user_id = row.user_id
            db.commit()
        expired = _verify(client, "expired@example.com", expired_code)
        assert expired.status_code == 400
        with SessionLocal() as db:
            row = db.scalar(select(AuthCode).where(AuthCode.user_id == code_user_id))
            assert row.attempts == 1

    with _client() as client:
        _signup(client, "reused@example.com")
        code = _code(fake_email[-1])
        assert _verify(client, "reused@example.com", code).status_code == 200
        reused = _verify(client, "reused@example.com", code)
        assert reused.status_code == 400


def test_resend_cooldown_and_five_per_hour(db_session, fake_email):
    with _client() as client:
        signup = _signup(client)
        user_id = db_session.scalar(select(User.id).where(User.email == "new@example.com"))
        assert signup.status_code == 201
        assert len(fake_email) == 1

        neutral = client.post(
            "/auth/resend-verification", json={"email": "new@example.com"}
        )
        assert neutral.status_code == 202
        assert len(fake_email) == 1

        for expected_count in range(2, 6):
            _age_latest_code(db_session, user_id)
            response = client.post(
                "/auth/resend-verification", json={"email": "new@example.com"}
            )
            assert response.status_code == 202
            assert len(fake_email) == expected_count

        _age_latest_code(db_session, user_id)
        response = client.post(
            "/auth/resend-verification", json={"email": "new@example.com"}
        )
        assert response.status_code == 202
        assert len(fake_email) == 5


def test_neutral_responses_and_no_code_in_responses_or_logs(
    db_session, fake_email, caplog
):
    with _client() as client:
        _signup(client)
        code = _code(fake_email[-1])
        existing_forgot = client.post(
            "/auth/forgot-password", json={"email": "new@example.com"}
        )
        missing_forgot = client.post(
            "/auth/forgot-password", json={"email": "missing@example.com"}
        )
        assert existing_forgot.status_code == missing_forgot.status_code == 202
        assert existing_forgot.json() == missing_forgot.json()
        assert code not in existing_forgot.text
        assert code not in missing_forgot.text

        _verify(client, "new@example.com", code)
        existing_resend = client.post(
            "/auth/resend-verification", json={"email": "new@example.com"}
        )
        missing_resend = client.post(
            "/auth/resend-verification", json={"email": "missing@example.com"}
        )
        assert existing_resend.status_code == missing_resend.status_code == 202
        assert existing_resend.json() == missing_resend.json()
        assert code not in existing_resend.text
        assert code not in missing_resend.text
    assert code not in caplog.text


def test_password_reset_invalidates_old_sessions_and_all_codes(
    db_session, fake_email
):
    with _client() as client:
        _signup(client)
        verify_code = _code(fake_email[-1])
        verified = _verify(client, "new@example.com", verify_code)
        assert verified.status_code == 200
        old_token = verified.json()["access_token"]

        with SessionLocal() as db:
            user = db.scalar(select(User).where(User.email == "new@example.com"))
            row = db.scalar(select(AuthCode).where(AuthCode.user_id == user.id))
            row.created_at = _utcnow() - timedelta(seconds=61)
            db.commit()

        neutral = client.post(
            "/auth/forgot-password", json={"email": "new@example.com"}
        )
        assert neutral.status_code == 202
        reset_code = _code(fake_email[-1])
        reset = client.post(
            "/auth/reset-password",
            json={
                "email": "new@example.com",
                "code": reset_code,
                "new_password": "updated-password",
            },
        )
        assert reset.status_code == 200
        assert "access_token" not in reset.json()
        assert reset_code not in reset.text
        assert client.get(
            "/auth/me", headers={"Authorization": f"Bearer {old_token}"}
        ).status_code == 401
        with SessionLocal() as db:
            user = db.scalar(select(User).where(User.email == "new@example.com"))
            assert user.email_verified is True
            assert user.password_changed_at is not None
            codes = db.scalars(select(AuthCode).where(AuthCode.user_id == user.id)).all()
            assert all(row.used_at is not None for row in codes)
        fresh = client.post(
            "/auth/login",
            json={"email": "new@example.com", "password": "updated-password"},
        )
        assert fresh.status_code == 200
        assert client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {fresh.json()['access_token']}"},
        ).status_code == 200


def test_daily_email_limit_returns_typed_503(db_session, monkeypatch):
    settings = get_settings().model_copy(update={"MAX_EMAILS_PER_DAY": 0})
    monkeypatch.setattr(email_service, "get_settings", lambda: settings)
    with pytest.raises(EmailSendError) as caught:
        email_service.send_email("person@example.com", "Subject", "Text", "<p>Text</p>")
    assert caught.value.http_status == 503
    assert "temporarily unavailable" in str(caught.value)
    assert db_session.scalar(select(EmailDailyUsage)) is None


def test_daily_email_limit_is_a_friendly_signup_503(db_session, monkeypatch):
    settings = get_settings().model_copy(
        update={"MAX_EMAILS_PER_DAY": 0, "AUTH_RATE_LIMIT_PER_MINUTE": 100}
    )
    monkeypatch.setattr(email_service, "get_settings", lambda: settings)
    monkeypatch.setattr(auth_router, "get_settings", lambda: settings)
    with _client() as client:
        response = _signup(client)
    assert response.status_code == 503
    assert response.json()["detail"] == email_service.EMAIL_LIMIT_MESSAGE
    assert "access_token" not in response.text


def test_gmail_request_retry_and_safe_errors(db_session, monkeypatch):
    settings = get_settings().model_copy(
        update={
            "EMAIL_BACKEND": "gmail",
            "GMAIL_CLIENT_ID": "client-id",
            "GMAIL_CLIENT_SECRET": "client-secret",
            "GMAIL_REFRESH_TOKEN": "refresh-token",
            "GMAIL_USER": "sender@gmail.com",
            "EMAIL_FROM_NAME": "Test Sender",
        }
    )
    monkeypatch.setattr(email_service, "get_settings", lambda: settings)
    calls = []
    token_calls = []

    def retry_handler(request):
        if request.url == email_service.GMAIL_TOKEN_URL:
            token_calls.append(request)
            return httpx.Response(200, json={"access_token": "test-token", "expires_in": 3600}, request=request)
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, text="provider body with secret", request=request)
        return httpx.Response(200, json={}, request=request)

    monkeypatch.setattr(
        email_service,
        "_build_client",
        lambda: httpx.Client(transport=httpx.MockTransport(retry_handler)),
    )
    email_service.send_email(
        "to@example.com", "Hello", "Plain text", "<p>HTML</p>"
    )
    assert len(token_calls) == 1
    assert len(calls) == 2
    request = calls[1]
    assert request.url == email_service.GMAIL_SEND_URL
    assert "Bearer" in request.headers["Authorization"]
    body = json.loads(request.content)
    assert "raw" in body
    assert isinstance(body["raw"], str)
    # Verify the raw field is a valid base64 string
    base64.urlsafe_b64decode(body["raw"] + "==")

    def failing_handler(request):
        if request.url == email_service.GMAIL_TOKEN_URL:
            return httpx.Response(200, json={"access_token": "test-token", "expires_in": 3600}, request=request)
        return httpx.Response(
            503, text="client-secret and provider response secret", request=request
        )

    monkeypatch.setattr(
        email_service,
        "_build_client",
        lambda: httpx.Client(transport=httpx.MockTransport(failing_handler)),
    )
    with pytest.raises(EmailSendError) as caught:
        email_service.send_email("to@example.com", "Hello", "Text", "<p>Text</p>")
    assert "client-secret" not in str(caught.value)
    assert "provider response secret" not in str(caught.value)

    network_calls = []

    def network_retry_handler(request):
        if request.url == email_service.GMAIL_TOKEN_URL:
            return httpx.Response(200, json={"access_token": "test-token", "expires_in": 3600}, request=request)
        network_calls.append(request)
        if len(network_calls) == 1:
            raise httpx.ConnectError("network detail with credential", request=request)
        return httpx.Response(200, request=request)

    monkeypatch.setattr(
        email_service,
        "_build_client",
        lambda: httpx.Client(transport=httpx.MockTransport(network_retry_handler)),
    )
    email_service.send_email("to@example.com", "Hello", "Text", "<p>Text</p>")
    assert len(network_calls) == 2


def test_gmail_missing_credentials_fail_during_startup(monkeypatch):
    original = get_settings()
    incomplete = original.model_copy(
        update={
            "EMAIL_BACKEND": "gmail",
            "GMAIL_CLIENT_ID": "",
            "GMAIL_CLIENT_SECRET": "",
            "GMAIL_REFRESH_TOKEN": "",
            "GMAIL_USER": "",
        }
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: incomplete)
    with pytest.raises(RuntimeError, match="GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET and GMAIL_REFRESH_TOKEN and GMAIL_USER"):
        with _client():
            pass


def test_every_auth_endpoint_uses_the_ip_rate_limiter(monkeypatch, fake_email):
    from app import rate_limit as rate_limit_module

    settings = get_settings().model_copy(update={"AUTH_RATE_LIMIT_PER_MINUTE": 1})
    monkeypatch.setattr(rate_limit_module, "get_settings", lambda: settings)
    requests = [
        ("/auth/signup", {"email": "limited@example.com", "password": "long-password"}),
        ("/auth/verify-email", {"email": "missing@example.com", "code": "000000"}),
        ("/auth/resend-verification", {"email": "missing@example.com"}),
        ("/auth/login", {"email": "missing@example.com", "password": "long-password"}),
        ("/auth/forgot-password", {"email": "missing@example.com"}),
        (
            "/auth/reset-password",
            {
                "email": "missing@example.com",
                "code": "000000",
                "new_password": "long-password",
            },
        ),
    ]
    with _client() as client:
        for path, body in requests:
            rate_limit.reset()
            first = client.post(path, json=body)
            assert first.status_code != 429, (path, first.text)
            second = client.post(path, json=body)
            assert second.status_code == 429, path


def test_cleanup_removes_old_unverified_accounts_and_old_inactive_codes(db_session):
    now = _utcnow()
    unverified = User(
        email="abandoned@example.com",
        password_hash="hash",
        email_verified=False,
        created_at=now - timedelta(hours=25),
    )
    verified = User(
        email="verified@example.com",
        password_hash="hash",
        email_verified=True,
        created_at=now - timedelta(days=5),
    )
    db_session.add_all([unverified, verified])
    db_session.flush()
    db_session.add_all(
        [
            AuthCode(
                user_id=unverified.id,
                purpose="verify",
                code_hash="a" * 64,
                expires_at=now - timedelta(days=2),
                created_at=now - timedelta(days=2),
                used_at=None,
            ),
            AuthCode(
                user_id=verified.id,
                purpose="reset",
                code_hash="b" * 64,
                expires_at=now - timedelta(days=2),
                created_at=now - timedelta(days=2),
                used_at=now - timedelta(days=2),
            ),
            AuthCode(
                user_id=verified.id,
                purpose="reset",
                code_hash="c" * 64,
                expires_at=now + timedelta(days=2),
                created_at=now - timedelta(days=2),
                used_at=None,
            ),
        ]
    )
    db_session.commit()
    unverified_id = unverified.id
    verified_id = verified.id
    deleted_users, deleted_codes = cleanup_auth_records()
    assert deleted_users == 1
    assert deleted_codes == 2
    db_session.expire_all()
    assert db_session.get(User, verified_id) is not None
    assert db_session.get(User, unverified_id) is None
    remaining = db_session.scalars(
        select(AuthCode).where(AuthCode.user_id == verified_id)
    ).all()
    assert len(remaining) == 1
    assert remaining[0].code_hash == "c" * 64


def test_admin_status_includes_email_service(db_session, monkeypatch):
    from app import security
    from app.rag.status import SERVICE_EMAIL, record_success

    settings = get_settings().model_copy(update={"ADMIN_EMAILS": "admin@example.com"})
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    with SessionLocal() as db:
        admin = User(
            email="admin@example.com",
            password_hash=hash_password("password-123"),
            email_verified=True,
        )
        db.add(admin)
        db.commit()
    record_success(SERVICE_EMAIL)
    token = security.create_access_token(admin.id)
    with _client() as client:
        response = client.get(
            "/admin/status", headers={"Authorization": f"Bearer {token}"}
        )
    assert response.status_code == 200
    assert response.json()["email"]["state"] == "ok"
    assert response.json()["email"]["sends_today"] == 1
