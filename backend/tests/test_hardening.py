"""Hardening tests: quotas, startup sweep, rate limiting, signup switch, CORS.

Everything is hermetic: a fake vector store stands in for PGVector via
monkeypatching, no real Groq/Jina/Neon calls are made, and every clock is
injectable, so no test sleeps.
"""

import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from starlette.requests import Request

from app import main as main_module
from app import maintenance as maintenance_module
from app import rate_limit
from app.config import get_settings
from app.deps import get_current_user
from app.main import create_app
from app.models import ChatMessage, ChatRole, ChatSession, Document, DocumentStatus, _utcnow
from app.rag import ingest as ingest_module
from app.rag.chat import NO_DOCUMENTS_MESSAGE
from app.routers import auth as auth_router
from app.routers import chat as chat_router
from app.routers import documents as documents_router

from .conftest import FakeVectorStore


def _settings_patch(monkeypatch, module, **update):
    patched = lambda: get_settings().model_copy(update=update)  # noqa: E731
    monkeypatch.setattr(module, "get_settings", patched)
    return patched


def _fake_delete(store):
    return lambda s, document_id, n: s.delete(ids=[f"{document_id}:{i}" for i in range(n)])


def _message_count(db_session):
    return db_session.scalar(select(func.count()).select_from(ChatMessage))


class TestStartupSweep:
    def test_fails_only_stale_processing_and_deletes_partial_vectors(self, db_session, users, monkeypatch):
        store = FakeVectorStore()
        monkeypatch.setattr(maintenance_module, "build_vectorstore", lambda embeddings=None: store)
        monkeypatch.setattr(ingest_module, "delete_document_vectors", _fake_delete(store))
        now = _utcnow()
        stale_processing = Document(
            id=uuid.uuid4(),
            user_id=users["a"].id,
            filename="stale.txt",
            status=DocumentStatus.PROCESSING,
            chunk_count=3,
            created_at=now - timedelta(hours=1),
        )
        recent_processing = Document(
            id=uuid.uuid4(),
            user_id=users["a"].id,
            filename="fresh.txt",
            status=DocumentStatus.PROCESSING,
            chunk_count=0,
            created_at=now,
        )
        old_ready = Document(
            id=uuid.uuid4(),
            user_id=users["a"].id,
            filename="ready.txt",
            status=DocumentStatus.READY,
            chunk_count=4,
            created_at=now - timedelta(days=5),
        )
        db_session.add_all([stale_processing, recent_processing, old_ready])
        db_session.commit()

        failed = maintenance_module.fail_stale_uploads()

        assert failed == 1
        db_session.expire_all()
        assert stale_processing.status == DocumentStatus.FAILED
        assert stale_processing.error == maintenance_module.STALE_UPLOAD_ERROR
        assert recent_processing.status == DocumentStatus.PROCESSING
        assert old_ready.status == DocumentStatus.READY
        assert store.deleted == [[f"{stale_processing.id}:{i}" for i in range(3)]]

    def test_default_settings_make_current_uploads_safe(self):
        assert get_settings().STALE_PROCESSING_MINUTES == 30

    def test_failing_sweep_does_not_prevent_boot(self, monkeypatch):
        def boom():
            raise RuntimeError("sweep exploded")

        monkeypatch.setattr(main_module, "fail_stale_uploads", boom)
        with TestClient(create_app()) as client:
            assert client.get("/health").json() == {"status": "ok"}


class TestDocumentCap:
    def test_rejects_before_reading_or_saving_the_file(self, db_session, users, monkeypatch):
        _settings_patch(monkeypatch, documents_router, MAX_DOCUMENTS_PER_USER=1)
        seeded = Document(id=uuid.uuid4(), user_id=users["a"].id, filename="taken.txt")
        db_session.add(seeded)
        db_session.commit()

        app = create_app()
        app.dependency_overrides[get_current_user] = lambda: users["a"]
        with TestClient(app) as client:
            # Unsupported suffix too: a 409 proves the quota check runs before
            # every other file check, byte reading included.
            response = client.post(
                "/documents",
                files={"file": ("report.xyz", b"binary", "application/octet-stream")},
            )
        assert response.status_code == 409
        assert response.json()["detail"] == "Document limit reached (1). Delete one to upload another."
        count = db_session.scalar(
            select(func.count()).select_from(Document).where(Document.user_id == users["a"].id)
        )
        assert count == 1

    def test_other_users_documents_do_not_count(self, db_session, users, monkeypatch):
        _settings_patch(monkeypatch, documents_router, MAX_DOCUMENTS_PER_USER=1)
        store = FakeVectorStore()
        monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
        monkeypatch.setattr(ingest_module, "delete_document_vectors", _fake_delete(store))
        theirs = Document(
            id=uuid.uuid4(),
            user_id=users["b"].id,
            filename="theirs.txt",
            status=DocumentStatus.READY,
            chunk_count=2,
        )
        db_session.add(theirs)
        db_session.commit()

        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 540, 790), "Integration PDF body")
        data = doc.tobytes()
        doc.close()

        app = create_app()
        app.dependency_overrides[get_current_user] = lambda: users["a"]
        with TestClient(app) as client:
            response = client.post(
                "/documents",
                files={"file": ("notes.pdf", data, "application/pdf")},
            )
        assert response.status_code == 202
        assert store.added  # ingestion actually ran for the upload


class TestDailyMessageCap:
    def _seed_message(self, db_session, owner, *, created_at):
        convo = ChatSession(
            id=uuid.uuid4(),
            user_id=owner.id,
            title="old chat",
            created_at=created_at,
            updated_at=created_at,
        )
        db_session.add(convo)
        db_session.add(
            ChatMessage(session_id=convo.id, role=ChatRole.USER, content="hi", created_at=created_at)
        )
        db_session.commit()
        return convo

    def test_blocks_at_limit_sync_and_stream(self, db_session, users, monkeypatch):
        _settings_patch(monkeypatch, chat_router, MAX_MESSAGES_PER_DAY=1)
        self._seed_message(db_session, users["a"], created_at=_utcnow() - timedelta(minutes=2))
        messages_before = _message_count(db_session)
        sessions_before = db_session.scalar(select(func.count()).select_from(ChatSession))

        app = create_app()
        app.dependency_overrides[get_current_user] = lambda: users["a"]
        with TestClient(app) as client:
            blocked = client.post("/chat", json={"question": "hi"})
            streamed = client.post("/chat/stream", json={"question": "hi"})

        assert blocked.status_code == 429
        assert blocked.json()["detail"] == "Daily message limit reached (1). Try again tomorrow."
        assert streamed.status_code == 429
        assert "application/json" in streamed.headers["content-type"]
        assert streamed.json()["detail"] == "Daily message limit reached (1). Try again tomorrow."
        assert _message_count(db_session) == messages_before
        assert db_session.scalar(select(func.count()).select_from(ChatSession)) == sessions_before

    def test_only_own_user_messages_within_window_count(self, db_session, users, monkeypatch):
        _settings_patch(monkeypatch, chat_router, MAX_MESSAGES_PER_DAY=5)
        now = _utcnow()
        for offset in range(5):
            self._seed_message(db_session, users["b"], created_at=now - timedelta(minutes=offset + 1))
        mine = self._seed_message(db_session, users["a"], created_at=now - timedelta(minutes=1))

        app = create_app()
        app.dependency_overrides[get_current_user] = lambda: users["a"]
        with TestClient(app) as client:
            cleared = client.post("/chat", json={"question": "what is up"})

        # B still has 5 recent messages, but a has 1: the cap counts only the
        # caller's own user turns. No ready documents reads the friendly reply.
        assert cleared.status_code == 200
        assert cleared.json()["answer"] == NO_DOCUMENTS_MESSAGE
        assert mine.id

    def test_messages_outside_window_do_not_count(self, db_session, users, monkeypatch):
        _settings_patch(monkeypatch, chat_router, MAX_MESSAGES_PER_DAY=1)
        self._seed_message(db_session, users["a"], created_at=_utcnow() - timedelta(hours=25))

        app = create_app()
        app.dependency_overrides[get_current_user] = lambda: users["a"]
        with TestClient(app) as client:
            response = client.post("/chat", json={"question": "fresh question"})

        assert response.status_code == 200
        assert response.json()["answer"] == NO_DOCUMENTS_MESSAGE


class TestAuthRateLimit:
    def _limit(self, monkeypatch, **update):
        return _settings_patch(monkeypatch, rate_limit, **update)

    def _request(self, headers=None, client=("198.51.100.7", 5555)):
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "raw_path": b"/",
                "query_string": b"",
                "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
                "client": client,
            }
        )

    def test_blocks_at_limit_with_retry_after(self, monkeypatch):
        self._limit(monkeypatch, AUTH_RATE_LIMIT_PER_MINUTE=2)
        assert rate_limit.allow("1.2.3.4", now=1000.0) == (True, 0)
        assert rate_limit.allow("1.2.3.4", now=1000.5) == (True, 0)
        blocked, retry_after = rate_limit.allow("1.2.3.4", now=1001.0)
        assert blocked is False
        assert retry_after == 59

    def test_attempt_is_counted_when_allowed(self, monkeypatch):
        self._limit(monkeypatch, AUTH_RATE_LIMIT_PER_MINUTE=1)
        assert rate_limit.allow("1.2.3.4", now=1000.0) == (True, 0)
        blocked, _ = rate_limit.allow("1.2.3.4", now=1001.0)
        assert blocked is False

    def test_resets_once_the_window_passes(self, monkeypatch):
        self._limit(monkeypatch, AUTH_RATE_LIMIT_PER_MINUTE=2)
        rate_limit.allow("1.2.3.4", now=1000.0)
        rate_limit.allow("1.2.3.4", now=1000.5)
        assert rate_limit.allow("1.2.3.4", now=1001.0) == (False, 59)
        allowed, _ = rate_limit.allow("1.2.3.4", now=1060.0)
        assert allowed is True

    def test_addresses_are_tracked_separately(self, monkeypatch):
        self._limit(monkeypatch, AUTH_RATE_LIMIT_PER_MINUTE=1)
        rate_limit.allow("1.2.3.4", now=1000.0)
        assert rate_limit.allow("2.3.4.5", now=1000.0) == (True, 0)
        assert rate_limit.allow("1.2.3.4", now=1000.5) == (False, 60)

    def test_socket_address_used_by_default(self, monkeypatch):
        self._limit(monkeypatch, AUTH_RATE_LIMIT_PER_MINUTE=1)
        spoofed = self._request(headers={"x-forwarded-for": "9.9.9.9, 8.8.8.8"})
        assert rate_limit.client_ip(spoofed) == "198.51.100.7"

    def test_rightmost_forwarded_entry_only_when_trusting_proxy(self, monkeypatch):
        self._limit(monkeypatch, TRUST_PROXY=True)
        forwarded = self._request(headers={"x-forwarded-for": "9.9.9.9, 8.8.8.8"})
        assert rate_limit.client_ip(forwarded) == "8.8.8.8"
        plain = self._request(headers={})
        assert rate_limit.client_ip(plain) == "198.51.100.7"

    def test_endpoint_returns_429_with_retry_after_header(self, monkeypatch):
        self._limit(monkeypatch, AUTH_RATE_LIMIT_PER_MINUTE=1)
        with TestClient(create_app()) as client:
            first = client.post("/auth/login", json={"email": "a@example.com", "password": "whatever1"})
            second = client.post("/auth/login", json={"email": "a@example.com", "password": "whatever1"})
        assert first.status_code in {401, 200}
        assert second.status_code == 429
        assert second.json()["detail"] == "Too many attempts. Please wait a minute and try again."
        assert int(second.headers["Retry-After"]) >= 1


class TestSignupToggle:
    def test_closed_signup_returns_403(self, monkeypatch):
        _settings_patch(monkeypatch, auth_router, SIGNUPS_ENABLED=False)
        with TestClient(create_app()) as client:
            response = client.post(
                "/auth/signup", json={"email": "new@example.com", "password": "longenough123"}
            )
        assert response.status_code == 403
        assert response.json()["detail"] == "New signups are currently closed."

    def test_open_signup_creates_the_user(self, monkeypatch):
        _settings_patch(monkeypatch, auth_router, SIGNUPS_ENABLED=True)
        with TestClient(create_app()) as client:
            response = client.post(
                "/auth/signup", json={"email": "fresh@example.com", "password": "longenough123"}
            )
        assert response.status_code == 201
        assert response.json()["user"]["email"] == "fresh@example.com"

    def test_config_endpoint_exposes_public_values(self, monkeypatch):
        _settings_patch(
            monkeypatch, main_module, SIGNUPS_ENABLED=True, RERANK_ENABLED=True, MAX_UPLOAD_MB=25
        )
        with TestClient(create_app()) as client:
            opened = client.get("/config")
        assert opened.status_code == 200
        assert opened.json() == {
            "signups_enabled": True,
            "max_upload_mb": 25,
            "allowed_extensions": [".pdf"],
            "rerank_available": True,
            "rerank_default": True,
        }

        _settings_patch(
            monkeypatch, main_module, SIGNUPS_ENABLED=False, RERANK_ENABLED=False, MAX_UPLOAD_MB=13
        )
        with TestClient(create_app()) as client:
            closed = client.get("/config")
        assert closed.json() == {
            "signups_enabled": False,
            "max_upload_mb": 13,
            "allowed_extensions": [".pdf"],
            "rerank_available": False,
            "rerank_default": False,
        }


class TestCorsOrigins:
    def _client(self, monkeypatch, **update):
        _settings_patch(monkeypatch, main_module, **update)
        return TestClient(create_app())

    def test_listed_origins_allowed_unlisted_rejected(self, monkeypatch):
        client = self._client(
            monkeypatch,
            FRONTEND_ORIGINS="https://a.example, https://b.example",
            FRONTEND_ORIGIN="https://should-not-count.example",
        )
        for path in ("/chat", "/documents"):
            allowed = client.options(
                path,
                headers={"Origin": "https://a.example", "Access-Control-Request-Method": "POST"},
            )
            assert allowed.headers.get("access-control-allow-origin") == "https://a.example"
            rejected = client.options(
                path,
                headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
            )
            assert "access-control-allow-origin" not in rejected.headers

    def test_wildcards_never_whitelist_everything(self, monkeypatch):
        client = self._client(
            monkeypatch,
            FRONTEND_ORIGINS="*, https://a.example",
            FRONTEND_ORIGIN="*",
        )
        assert client.options(
            "/chat",
            headers={"Origin": "https://a.example", "Access-Control-Request-Method": "POST"},
        ).headers.get("access-control-allow-origin") == "https://a.example"
        any_origin = client.options(
            "/chat",
            headers={"Origin": "https://random.example", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in any_origin.headers

    def test_empty_list_falls_back_to_single_origin(self, monkeypatch):
        client = self._client(
            monkeypatch,
            FRONTEND_ORIGINS="",
            FRONTEND_ORIGIN="http://localhost:3000",
        )
        assert client.options(
            "/chat",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
        ).headers.get("access-control-allow-origin") == "http://localhost:3000"
        stealth = client.options(
            "/chat",
            headers={"Origin": "https://other.example", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in stealth.headers
