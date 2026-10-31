"""Service status tracking + admin status API tests.

Everything is fake: Jina is driven over httpx.MockTransport (both the
embeddings client and the admin status probe), Groq is a fake BaseChatModel,
and the database is in-process SQLite. No real Jina, Groq, or Neon calls.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.deps import get_current_user
from app.main import create_app
from app.models import (
    ChatMessage,
    ChatRole,
    ChatSession,
    DocumentStatus,
    ServiceStatus,
    ServiceUsage,
)
from app.rag import retrieve as retrieve_module
from app.rag import status as status_module
from app.rag.embeddings import JinaEmbeddingError, JinaEmbeddings
from app.rag.rerank import JinaRerankError, JinaReranker
from app.rag.status import (
    SERVICE_GROQ,
    SERVICE_JINA,
    SERVICE_UNAVAILABLE_MESSAGE,
    ServiceUnavailableError,
    classify_failure,
    record_failure,
    record_success,
)
from app.routers import admin as admin_router
from app.routers import chat as chat_router
from app.security import is_admin_email

from .test_chat import _parse_sse

_EMBEDDING = [0.125] * 1024  # matches the default EMBEDDING_DIMENSION


def _jina_handler(*, status_code: int = 200, tokens: int | None = None, body: str | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        payload_body = body
        if status_code != 200 and payload_body is None:
            payload_body = '{"detail": "rejected"}'
        if status_code != 200:
            return httpx.Response(status_code, content=payload_body.encode("utf-8"), request=request)
        payload = {"data": [{"index": 0, "embedding": _EMBEDDING}]}
        if tokens is not None:
            payload["usage"] = {"total_tokens": tokens}
        return httpx.Response(200, json=payload, request=request)

    return handler


def _embeddings_client(monkeypatch, handler, api_key="jina_probe_key") -> JinaEmbeddings:
    monkeypatch.setattr(JinaEmbeddings, "_throttle", lambda self: None)
    client = JinaEmbeddings(api_key=api_key, model="m", dimensions=1024)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


class _FakeLLM(BaseChatModel):
    reply: str = "fake answer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="hello "))

    @property
    def _llm_type(self) -> str:
        return "fake-record"


class _FakeRateLimitError(Exception):
    status_code = 429


class _UserStore:
    """Single matching chunk for the filtered owner, like PGVector would."""

    def similarity_search(self, query, k=4, filter=None, **kwargs):
        owner = (filter or {}).get("user_id", "")
        if not owner:
            return []
        return [
            Document(
                page_content="cats live here",
                metadata={"user_id": owner, "document_id": "d1", "source": "d1.txt", "chunk_index": 0},
            )
        ]


@pytest.fixture()
def ready_chat_store(monkeypatch, users, document):
    """The caller has a READY document; retrieval and the LLM are faked."""
    document.status = DocumentStatus.READY
    document.chunk_count = 1
    users["session"].commit()
    store = _UserStore()
    monkeypatch.setattr(retrieve_module, "build_vectorstore", lambda: store)
    monkeypatch.setattr(chat_router, "build_llm", lambda: _FakeLLM())
    return store


@pytest.fixture()
def no_admin_emails():
    """Default settings: an empty admin list, so nobody is an admin."""
    settings = get_settings()
    with_admins = settings.ADMIN_EMAILS
    settings.ADMIN_EMAILS = ""
    try:
        yield settings
    finally:
        settings.ADMIN_EMAILS = with_admins


@pytest.fixture()
def make_admin(monkeypatch):
    """List the given user's (exact, lowercase) email in ADMIN_EMAILS."""

    def _make(user):
        monkeypatch.setattr(get_settings(), "ADMIN_EMAILS", user.email.lower())
        return user

    return _make


@pytest.fixture()
def check_cooldown_cleared():
    admin_router.reset_check_cooldown()
    yield
    admin_router.reset_check_cooldown()


def _app_for(user) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: user
    return TestClient(app)


def _status_row(service: str):
    with SessionLocal() as db:
        return db.get(ServiceStatus, service)


def _usage_row(service: str):
    day = datetime.now(timezone.utc).date()
    with SessionLocal() as db:
        return db.scalar(
            select(ServiceUsage).where(ServiceUsage.service == service, ServiceUsage.day == day)
        )


class TestClassification:
    def test_jina_classification_table(self):
        cases = [
            (401, "unauthorized", "invalid_key"),
            (403, "forbidden", "invalid_key"),
            (402, "payment required", "exhausted"),
            (429, "slow down", "rate_limited"),
            (400, '{"detail": "device balance is too low"}', "exhausted"),
            (400, '"device quota exceeded"', "exhausted"),
            (400, '"insufficient credits for this key"', "exhausted"),
            (400, "bad model name or task", "error"),
            (400, "insufficient", "exhausted"),
            (500, "internal error", "unavailable"),
            (503, "maintenance", "unavailable"),
            (None, "connection reset", "unavailable"),  # network error
        ]
        for code, body, expected in cases:
            state, message = classify_failure(SERVICE_JINA, code, body)
            assert state == expected, f"{code}/{body!r} -> {state}"
            assert message  # always a sanitized message

    def test_groq_classification_table(self):
        assert classify_failure(SERVICE_GROQ, 401, "bad key")[0] == "invalid_key"
        state, message = classify_failure(SERVICE_GROQ, 429, "slow down", retry_after="17")
        assert state == "rate_limited"
        assert "17" in message
        assert classify_failure(SERVICE_GROQ, 500, "boom")[0] == "unavailable"
        assert classify_failure(SERVICE_GROQ, None, "timeout")[0] == "unavailable"
        assert classify_failure(SERVICE_GROQ, 400, "bad request")[0] == "error"

    def test_unknown_service_rejected(self):
        with pytest.raises(ValueError):
            classify_failure("openai", 500, "boom")

    def test_message_sanitized_and_truncated(self):
        secret_body = '401 unauthorized for Bearer jina_9f3a7b2cXYZ and gsk_abc123def key'
        _state, message = classify_failure(SERVICE_JINA, 401, secret_body)
        assert "jina_9f3a7b2cXYZ" not in message
        assert "gsk_abc123def" not in message
        assert "Bearer" not in message
        assert "[REDACTED]" in message
        assert " ".join(message.split()) == message
        _state, long_message = classify_failure(
            SERVICE_JINA, 500, "x" * 250
        )
        assert len(long_message) <= 200

    def test_service_unavailable_message_is_client_safe(self):
        assert "jina" not in SERVICE_UNAVAILABLE_MESSAGE.lower()
        assert "key" not in SERVICE_UNAVAILABLE_MESSAGE.lower()


class TestRecorderWrites:
    def test_success_then_failure_then_success_latest_outcome_wins(self, db_session):
        record_success(SERVICE_JINA, tokens=5)
        row = _usage_row(SERVICE_JINA)
        assert row is not None
        assert row.requests == 1 and row.tokens == 5 and row.errors == 0

        record_failure(SERVICE_JINA, 500, "boom")
        status_row = _status_row(SERVICE_JINA)
        assert status_row.state == "unavailable"
        assert status_row.http_status == 500
        assert status_row.last_error_at is not None

        record_success(SERVICE_JINA, tokens=7)
        status_row = _status_row(SERVICE_JINA)
        assert status_row.state == "ok"
        assert status_row.message is None
        assert status_row.last_success_at is not None
        assert status_row.last_error_at is not None  # history kept
        usage = _usage_row(SERVICE_JINA)
        # Failures count as requests too; tokens only accrue on success.
        assert usage.requests == 3 and usage.tokens == 12 and usage.errors == 1

    def test_recorder_failure_never_raises_or_breaks_the_caller(self, db_session, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("db exploded")

        monkeypatch.setattr(status_module, "SessionLocal", explode)
        record_success(SERVICE_JINA, tokens=5)  # must not raise
        record_failure(SERVICE_JINA, 500, "boom")  # must not raise

    def test_unknown_when_no_calls_since_boot(self, db_session):
        assert _status_row(SERVICE_JINA) is None

    def test_groq_429_keeps_retry_after_in_message(self, db_session):
        record_failure(SERVICE_GROQ, 429, "rate limited", retry_after="12")
        row = _status_row(SERVICE_GROQ)
        assert row.state == "rate_limited"
        assert "12" in row.message


class TestEmbeddingsRecording:
    def test_tokens_summed_from_successful_responses(self, db_session, monkeypatch):
        client = _embeddings_client(monkeypatch, _jina_handler(tokens=5))
        client.embed_query("hello")
        client.embed_query("world")  # second call with the same handler

        usage = _usage_row(SERVICE_JINA)
        assert usage.requests == 2
        assert usage.tokens == 10
        assert usage.errors == 0
        status_row = _status_row(SERVICE_JINA)
        assert status_row.state == "ok"

    def test_402_and_401_become_typed_unavailable_and_recorded(self, db_session, monkeypatch):
        client = _embeddings_client(monkeypatch, _jina_handler(status_code=402))
        with pytest.raises(ServiceUnavailableError) as info:
            client.embed_query("hello")
        assert str(info.value) == SERVICE_UNAVAILABLE_MESSAGE
        assert _status_row(SERVICE_JINA).state == "exhausted"

        client = _embeddings_client(monkeypatch, _jina_handler(status_code=401))
        with pytest.raises(ServiceUnavailableError):
            client.embed_query("hello")
        row = _status_row(SERVICE_JINA)
        assert row.state == "invalid_key"
        assert row.http_status == 401

    def test_retries_exhausted_record_final_failure(self, db_session, monkeypatch):
        monkeypatch.setattr(
            JinaEmbeddings,
            "_sleep_backoff",
            lambda self, attempt, retry_after=None: None,
        )
        client = _embeddings_client(monkeypatch, _jina_handler(status_code=500))
        with pytest.raises(JinaEmbeddingError):
            client.embed_query("hello")
        row = _status_row(SERVICE_JINA)
        assert row.state == "unavailable"
        assert row.http_status == 500
        usage = _usage_row(SERVICE_JINA)
        assert usage.errors == 1 and usage.requests == 1

    def test_network_error_classified_unavailable(self, db_session, monkeypatch):
        monkeypatch.setattr(
            JinaEmbeddings,
            "_sleep_backoff",
            lambda self, attempt, retry_after=None: None,
        )

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        client = _embeddings_client(monkeypatch, handler)
        with pytest.raises(JinaEmbeddingError):
            client.embed_query("hello")
        assert _status_row(SERVICE_JINA).state == "unavailable"

    def test_recorder_write_failure_never_breaks_the_embedding(
        self, monkeypatch
    ):
        def explode(*args, **kwargs):
            raise RuntimeError("recorder is broken")

        monkeypatch.setattr(status_module, "record_success", explode)
        monkeypatch.setattr(status_module, "record_failure", explode)
        client = _embeddings_client(monkeypatch, _jina_handler(tokens=9))
        vectors = client.embed_query("hello")
        assert len(vectors) == 1024


class TestRerankRecording:
    def test_success_tokens_recorded(self, db_session, monkeypatch):
        def rerank_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"results": [{"index": 0, "relevance_score": 0.5}], "usage": {"total_tokens": 4}},
                request=request,
            )

        reranker = JinaReranker(api_key="k", model="m", client=httpx.Client(transport=httpx.MockTransport(rerank_handler)))
        monkeypatch.setattr(JinaReranker, "_throttle", lambda self: None)
        scored = reranker.rerank("q", ["doc one", "doc two"])
        assert scored[0][0] == 0
        usage = _usage_row(SERVICE_JINA)
        assert usage.tokens == 4 and usage.requests == 1


class TestChatRecording:
    @staticmethod
    def _post_chat(client: TestClient, path: str):
        return client.post(path, json={"question": "cats"})

    def test_groq_success_recorded_on_stream(self, ready_chat_store, users, monkeypatch, db_session):
        with _app_for(users["a"]) as client:
            response = self._post_chat(client, "/chat/stream")
        assert response.status_code == 200
        events = _parse_sse(response.text)
        names = [name for name, _data in events]
        assert names == ["session", "sources", "token", "done"]
        usage = _usage_row(SERVICE_GROQ)
        assert usage is not None and usage.requests == 1
        assert _status_row(SERVICE_GROQ).state == "ok"

    def test_groq_rate_limit_recorded_but_error_unchanged(
        self, ready_chat_store, users, monkeypatch, db_session
    ):
        class _FailLLM(_FakeLLM):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                raise _FakeRateLimitError("groq says: slow down, internal detail")

            async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
                raise _FakeRateLimitError("groq says: slow down, internal detail")
                yield ChatGenerationChunk(message=AIMessageChunk(content=""))

        monkeypatch.setattr(chat_router, "build_llm", lambda: _FailLLM())
        with _app_for(users["a"]) as client:
            response = self._post_chat(client, "/chat")
        assert response.status_code == 502
        assert response.json()["detail"] == (
            "The answering service is a bit overloaded right now. Please try again in a few seconds."
        )
        assert "slow down" not in response.text
        row = _status_row(SERVICE_GROQ)
        assert row.state == "rate_limited"
        assert row.http_status == 429

    def test_recorder_failure_never_breaks_chat_request(
        self, ready_chat_store, users, monkeypatch, db_session
    ):
        def explode(*args, **kwargs):
            raise RuntimeError("recorder is broken")

        monkeypatch.setattr(chat_router, "record_success", explode)
        with _app_for(users["a"]) as client:
            response = self._post_chat(client, "/chat")
        assert response.status_code == 200
        assert response.json()["answer"] == "fake answer"

    def test_stream_service_unavailable_error_event(self, ready_chat_store, users, monkeypatch, db_session):
        def unavailable(*args, **kwargs):
            raise ServiceUnavailableError()

        monkeypatch.setattr(chat_router, "retrieve_reranked_result", unavailable)
        with _app_for(users["a"]) as client:
            response = self._post_chat(client, "/chat/stream")
        assert response.status_code == 200
        events = _parse_sse(response.text)
        names = [name for name, _data in events]
        # Event order unchanged: session first, then the error event in the
        # position where the generic retrieval error would appear.
        assert names == ["session", "error"]
        payload = json.loads(events[1][1])
        assert payload["error"] == SERVICE_UNAVAILABLE_MESSAGE
        assert payload["code"] == "service_unavailable"
        assert set(payload) == {"error", "code"}

    def test_pre_stream_503_when_jina_exhausted(self, users, db_session):
        with SessionLocal() as db:
            db.add(
                ServiceStatus(
                    service=SERVICE_JINA, state="exhausted", http_status=402, message="quota gone"
                )
            )
            db.commit()
        with _app_for(users["a"]) as client:
            stream = client.post("/chat/stream", json={"question": "cats"})
            sync = client.post("/chat", json={"question": "cats"})
        assert stream.status_code == sync.status_code == 503
        assert stream.json()["detail"] == SERVICE_UNAVAILABLE_MESSAGE
        # The friendly 503 must carry no provider details.
        assert "jina" not in stream.text.lower() and "402" not in stream.text


class TestAdminAuth:
    def test_non_admin_gets_404_on_all_three_routes(self, users, no_admin_emails):
        with _app_for(users["b"]) as client:
            assert client.get("/admin/status").status_code == 404
            assert client.post("/admin/status/check").status_code == 404
            assert client.post("/admin/status/reset-usage").status_code == 404

    def test_is_admin_computed_for_different_email_casing(self, users, db_session, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "ADMIN_EMAILS", "Admin@Example.COM,other@x.io")
        assert is_admin_email("admin@example.com") is True
        assert is_admin_email("ADMIN@EXAMPLE.COM") is True
        assert is_admin_email("someone@example.com") is False
        assert is_admin_email("") is False

        # And through the API: /auth/me reflects the server-side computation.
        admin_user = users["a"]  # a@example.com must be case-insensitively listed
        users["a"].email = "admin@example.com"
        with _app_for(admin_user) as client:
            body = client.get("/auth/me").json()
        assert body["is_admin"] is True
        assert set(body) == {"id", "email", "created_at", "is_admin"}

        users["b"].email = "b@example.com"
        with _app_for(users["b"]) as client:
            assert client.get("/auth/me").json()["is_admin"] is False
        # The listed (case-adjusted) user passes the admin gate.
        with _app_for(admin_user) as client:
            assert client.get("/admin/status").status_code == 200

    def test_admin_email_list_is_case_insensitive_through_signup(self, users, no_admin_emails, db_session, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "ADMIN_EMAILS", "Chief@Example.Com")
        app = create_app()
        with TestClient(app) as client:
            response = client.post(
                "/auth/signup",
                json={"email": "Chief@Example.Com", "password": "longenough8"},
            )
        assert response.status_code == 201
        assert response.json()["user"]["is_admin"] is True
        assert response.json()["user"]["email"] == "chief@example.com"


class TestAdminStatusPayload:
    def test_payload_shape_and_unknown_state(self, users, db_session, make_admin):
        with _app_for(make_admin(users["a"])) as client:
            body = client.get("/admin/status")
        assert body.status_code == 200
        payload = body.json()
        assert set(payload) == {"jina", "groq", "checked_at"}
        assert payload["jina"]["state"] == "unknown"
        assert set(payload["jina"]) == {
            "state", "http_status", "message", "last_success_at", "last_error_at",
            "tokens_used_since_reset", "tokens_used_today", "requests_today",
            "errors_today", "tokens_budget", "tokens_estimated_remaining",
            "percent_remaining", "low",
        }
        assert set(payload["groq"]) == {
            "state", "message", "last_success_at", "last_error_at", "answers_last_24h",
        }
        assert payload["groq"]["answers_last_24h"] == 0
        # No API keys or key fragments anywhere in the response.
        assert "test-jina-key" not in body.text
        assert "test-groq-key" not in body.text

    def test_answers_last_24h_counts_all_users(self, users, db_session, make_admin):
        with SessionLocal() as db:
            for owner in (users["a"].id, users["b"].id):
                session_row = ChatSession(id=uuid.uuid4(), user_id=owner, title="t")
                db.add(session_row)
                db.flush()
                db.add(
                    ChatMessage(
                        session_id=session_row.id, role=ChatRole.ASSISTANT, content="fresh answer"
                    )
                )
                db.add(
                    ChatMessage(
                        session_id=session_row.id,
                        role=ChatRole.ASSISTANT,
                        content="old answer",
                        created_at=datetime.now(timezone.utc) - timedelta(hours=25),
                    )
                )
            db.commit()
        with _app_for(make_admin(users["a"])) as client:
            payload = client.get("/admin/status").json()
        assert payload["groq"]["answers_last_24h"] == 2

    def test_estimate_floored_at_zero_and_low_flag(self, users, db_session, make_admin, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "JINA_TOKEN_BUDGET", 100)
        record_success(SERVICE_JINA, tokens=250)
        with _app_for(make_admin(users["a"])) as client:
            payload = client.get("/admin/status").json()["jina"]
        assert payload["tokens_used_since_reset"] == 250
        assert payload["tokens_budget"] == 100
        assert payload["tokens_estimated_remaining"] == 0
        assert payload["percent_remaining"] == 0
        assert payload["low"] is True

    def test_reset_usage_zeroes_the_since_reset_counter(
        self, users, db_session, make_admin, check_cooldown_cleared
    ):
        record_success(SERVICE_JINA, tokens=500)
        with _app_for(make_admin(users["a"])) as client:
            before = client.get("/admin/status").json()["jina"]
            assert before["tokens_used_since_reset"] == 500
            assert before["tokens_estimated_remaining"] == before["tokens_budget"] - 500
            reset = client.post("/admin/status/reset-usage")
            after = reset.json()["jina"]
        assert reset.status_code == 200
        assert after["tokens_used_since_reset"] == 0
        assert after["tokens_used_today"] == 0
        assert after["tokens_estimated_remaining"] == after["tokens_budget"]
        assert after["percent_remaining"] == 100
        assert after["low"] is False


class TestAdminCheck:
    def _mocked_check(self, monkeypatch, *, status_code: int = 200, tokens: int | None = None):
        handled = {"saw_groq": False}

        def handler(request: httpx.Request) -> httpx.Response:
            if "api.groq.com" in str(request.url):
                handled["saw_groq"] = True
            if status_code != 200:
                return httpx.Response(status_code, content="bad", request=request)
            return httpx.Response(
                200,
                json={
                    "data": [{"index": 0, "embedding": _EMBEDDING[:4]}],
                    "usage": {"total_tokens": tokens},
                },
                request=request,
            )

        def build_check_client():
            return httpx.Client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(status_module, "build_check_client", build_check_client)
        return handled

    def test_check_makes_one_jina_call_records_and_return_payload(
        self, users, db_session, make_admin, check_cooldown_cleared, monkeypatch
    ):
        admin = make_admin(users["a"])
        handled = self._mocked_check(monkeypatch, tokens=11)
        with _app_for(admin) as client:
            response = client.post("/admin/status/check")
        assert response.status_code == 200
        payload = response.json()
        assert payload["jina"]["state"] == "ok"
        assert payload["jina"]["tokens_used_today"] == 11
        assert payload["jina"]["requests_today"] == 1
        assert handled["saw_groq"] is False  # must never call Groq

    def test_check_cooldown_enforced_with_retry_after(
        self, users, db_session, make_admin, check_cooldown_cleared, monkeypatch
    ):
        admin = make_admin(users["a"])
        self._mocked_check(monkeypatch, tokens=3)
        with _app_for(admin) as client:
            first = client.post("/admin/status/check")
            assert first.status_code == 200
            second = client.post("/admin/status/check")
            assert second.status_code == 429
            assert int(second.headers["Retry-After"]) >= 1

            # Once the cooldown window has passed the check works again.
            cooldown = get_settings().STATUS_CHECK_COOLDOWN_SECONDS
            admin_router._last_check_monotonic -= cooldown + 1
            third = client.post("/admin/status/check")
            assert third.status_code == 200

    def test_failed_check_records_http_status(
        self, users, db_session, make_admin, check_cooldown_cleared, monkeypatch
    ):
        admin = make_admin(users["a"])
        self._mocked_check(monkeypatch, status_code=401)
        with _app_for(admin) as client:
            payload = client.post("/admin/status/check").json()
        assert payload["jina"]["state"] == "invalid_key"
        assert payload["jina"]["http_status"] == 401

    def test_network_error_check_records_unavailable(
        self, users, db_session, make_admin, check_cooldown_cleared, monkeypatch
    ):
        admin = make_admin(users["a"])

        def build_check_client():
            def handler(request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("down", request=request)

            return httpx.Client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(status_module, "build_check_client", build_check_client)
        with _app_for(admin) as client:
            payload = client.post("/admin/status/check").json()
        assert payload["jina"]["state"] == "unavailable"

    def test_check_responses_never_contain_keys(
        self, users, db_session, make_admin, check_cooldown_cleared, monkeypatch
    ):
        admin = make_admin(users["a"])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                content='{"detail": "bad Bearer test-jina-key ok"}',
                request=request,
            )

        monkeypatch.setattr(
            status_module, "build_check_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
        )
        with _app_for(admin) as client:
            response = client.post("/admin/status/check")
        assert response.status_code == 200
        assert "test-jina-key" not in response.text
        assert "[REDACTED]" in response.json()["jina"]["message"]
