"""Chat endpoint tests. Everything is fake: no real Groq, Jina, or Neon calls.

The fake vector store mirrors PGVector's contract: similarity_search(query,
k, filter) must honour the metadata filter, exactly like langchain_postgres
applies it inside the SQL query. That lets the tests prove the caller's
user_id is always in the filter and that other users' chunks never surface.
"""

import json
import uuid
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.deps import get_current_user
from app.main import create_app
from app.models import Document as DocumentRow
from app.models import DocumentStatus
from app.rag import retrieve as retrieve_module
from app.rag.chat import (
    AUTH_ERROR_MESSAGE,
    GENERIC_ERROR_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    NO_RESULTS_MESSAGE,
    RATE_LIMIT_MESSAGE,
    friendly_error,
)
from app.rag.retrieve import retrieve_chunks
from app.routers import chat as chat_router


class FakeChatLLM(BaseChatModel):
    """Returns a canned answer and records every prompt it receives."""

    reply: str = "This is the fake answer."
    calls: ClassVar[list] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    @property
    def _llm_type(self) -> str:
        return "fake"


class FakeStreamLLM(BaseChatModel):
    """Streams a fixed sequence of text deltas and records calls."""

    calls: ClassVar[list] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("non-streaming generation should not be used here")

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        for piece in ("hello ", "from ", "the ", "fake ", "stream"):
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))

    @property
    def _llm_type(self) -> str:
        return "fake-stream"


# Unannotated class attribute: plain holder, not a pydantic field.
class StreamFailLLM(BaseChatModel):
    error_to_raise: ClassVar[BaseException | None] = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("non-streaming generation should not be used here")

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        raise StreamFailLLM.error_to_raise
        yield ChatGenerationChunk(message=AIMessageChunk(content=""))  # both methods must be async generators

    @property
    def _llm_type(self) -> str:
        return "fake-stream-fail"


class FakeRateLimitError(Exception):
    status_code = 429


class FakeAuthError(Exception):
    status_code = 401


class UserFilteredStore:
    """Fake PGVector: only hands back docs matching the filter's user_id."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.filters: list[dict] = []
        self.queries: list[str] = []

    def similarity_search(self, query, k=4, filter=None, **kwargs):
        self.queries.append(query)
        self.filters.append(dict(filter or {}))
        owner = (filter or {}).get("user_id", "")
        matches = [
            document
            for document in self.documents
            if document.metadata.get("user_id") == owner
        ]
        return matches[:k]


def _chunk(user_id: str, text: str, document_id: str, index: int = 0) -> Document:
    return Document(
        page_content=text,
        metadata={
            "user_id": user_id,
            "document_id": document_id,
            "source": f"{document_id}.txt",
            "chunk_index": index,
        },
    )


def _add_ready_document(session, user_id, name="only-b.txt"):
    row = DocumentRow(
        id=uuid.uuid4(),
        user_id=user_id,
        filename=name,
        status=DocumentStatus.READY,
        chunk_count=3,
    )
    session.add(row)
    session.commit()
    return row


def _parse_sse(text: str) -> list[tuple[str, str]]:
    """[(event, data_string), ...] in the exact order received."""
    events: list[tuple[str, str]] = []
    current_event = None
    for line in text.splitlines():
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ").strip()
        elif line.startswith("data: ") and current_event is not None:
            events.append((current_event, line.removeprefix("data: ")))
            current_event = None
    return events


@pytest.fixture()
def no_ready_store(monkeypatch, users, document):
    """Caller has a PROCESSING doc; only user B has READY documents."""
    _add_ready_document(users["session"], users["b"].id)
    store = UserFilteredStore([_chunk(str(users["b"].id), "secret b text", "b-doc")])
    monkeypatch.setattr(retrieve_module, "build_vectorstore", lambda: store)
    return store


@pytest.fixture()
def ready_store(monkeypatch, users, document):
    document.status = DocumentStatus.READY
    document.chunk_count = 2
    users["session"].commit()
    store = UserFilteredStore(
        [
            _chunk(str(users["a"].id), "own text about cats " * 20, "a-doc", 0),
            _chunk(str(users["a"].id), "second chunk about cats", "a-doc", 1),
            _chunk(str(users["b"].id), "secret text of user b", "b-doc", 0),
        ]
    )
    monkeypatch.setattr(retrieve_module, "build_vectorstore", lambda: store)
    return store


def _client(monkeypatch, user, llm):
    monkeypatch.setattr(chat_router, "build_llm", lambda: llm)
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: user
    return TestClient(app)


class TestRetrievalFilter:
    def test_filter_always_carries_caller_user_id(self, ready_store, users):
        retrieve_chunks("cats", users["a"].id, store=ready_store)
        assert ready_store.queries == ["cats"]
        assert ready_store.filters == [{"user_id": str(users["a"].id)}]

    def test_retrieval_returns_only_caller_chunks(self, ready_store, users):
        chunks = retrieve_chunks("cats", users["a"].id, store=ready_store)
        assert {chunk.text for chunk in chunks} == {
            "own text about cats " * 20,
            "second chunk about cats",
        }
        for chunk in chunks:
            assert chunk.document_id == "a-doc"
            assert chunk.source == "a-doc.txt"
            assert chunk.chunk_index in (0, 1)

    def test_other_user_gets_only_their_chunks(self, ready_store, users):
        chunks = retrieve_chunks("cats", users["b"].id, store=ready_store)
        assert ready_store.filters[-1] == {"user_id": str(users["b"].id)}
        assert len(chunks) == 1
        assert chunks[0].text == "secret text of user b"


class TestChatEndpoint:
    def test_answer_with_sources(self, monkeypatch, users, ready_store):
        FakeChatLLM.calls = []
        with _client(
            monkeypatch, users["a"], FakeChatLLM(reply="Cats rule, per documents.")
        ) as sync_client:
            response = sync_client.post("/chat", json={"question": "  what about cats?  "})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Cats rule, per documents."
        assert FakeChatLLM.calls, "the LLM chain was not invoked"
        prompt_text = str(FakeChatLLM.calls[0][1].content)
        assert "what about cats?" in prompt_text
        assert "own text about cats" in prompt_text

        sources = body["sources"]
        assert sources == [
            {
                "document_id": "a-doc",
                "source": "a-doc.txt",
                "chunk_index": 0,
                "snippet": ("own text about cats " * 20)[:200] + "...",
            },
            {
                "document_id": "a-doc",
                "source": "a-doc.txt",
                "chunk_index": 1,
                "snippet": "second chunk about cats",
            },
        ]

    def test_no_ready_documents_skips_llm_and_retrieval(
        self, monkeypatch, users, no_ready_store
    ):
        FakeChatLLM.calls = []
        with _client(monkeypatch, users["a"], FakeChatLLM(reply="should not happen")) as sync_client:
            response = sync_client.post("/chat", json={"question": "what about cats?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == NO_DOCUMENTS_MESSAGE
        assert body["sources"] == []
        assert not FakeChatLLM.calls, "the LLM must not be called without ready documents"
        assert no_ready_store.queries == [], "retrieval must not run either"

    def test_empty_retrieval_skips_llm(self, monkeypatch, users, ready_store):
        ready_store.documents = [_chunk("someone-else", "not the caller's text", "b-doc")]
        FakeChatLLM.calls = []
        with _client(monkeypatch, users["a"], FakeChatLLM(reply="nope")) as sync_client:
            response = sync_client.post("/chat", json={"question": "cats?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == NO_RESULTS_MESSAGE
        assert body["sources"] == []
        assert not FakeChatLLM.calls

    def test_question_validation(self, monkeypatch, users, ready_store):
        with _client(monkeypatch, users["a"], FakeChatLLM()) as sync_client:
            empty = sync_client.post("/chat", json={"question": "   "})
            toolong = sync_client.post("/chat", json={"question": "x" * 2001})
        assert empty.status_code == 422
        assert "empty" in empty.json()["detail"]
        assert toolong.status_code == 422
        assert "2000" in toolong.json()["detail"]

    def test_rate_limit_becomes_friendly_message(self, monkeypatch, users, ready_store):
        ExplodingSyncLLM.error_to_raise = FakeRateLimitError("groq says: slow down")
        with _client(monkeypatch, users["a"], ExplodingSyncLLM()) as sync_client:
            response = sync_client.post("/chat", json={"question": "cats"})
        assert response.status_code == 502
        assert response.json()["detail"] == RATE_LIMIT_MESSAGE
        assert "confidential" not in response.text


class ExplodingSyncLLM(BaseChatModel):
    error_to_raise: ClassVar[BaseException | None] = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise ExplodingSyncLLM.error_to_raise

    @property
    def _llm_type(self) -> str:
        return "fake-exploding"


class TestChatStreamEndpoint:
    def test_event_order_sources_tokens_done(self, monkeypatch, users, ready_store):
        FakeStreamLLM.calls = []
        with _client(monkeypatch, users["a"], FakeStreamLLM()) as stream_client:
            response = stream_client.post("/chat/stream", json={"question": "cats?"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache, no-transform"
        assert response.headers["x-accel-buffering"] == "no"

        events = _parse_sse(response.text)
        names = [name for name, data in events]
        assert names[0] == "sources"
        assert names[-1] == "done"
        assert set(names[1:-1]) == {"token"}
        assert "error" not in names
        assert FakeStreamLLM.calls, "the streaming chain was not invoked"
        assert "cats?" in str(FakeStreamLLM.calls[0][1].content)

    def test_sources_sent_first_as_json_without_other_users(
        self, monkeypatch, users, ready_store
    ):
        with _client(monkeypatch, users["a"], FakeStreamLLM()) as stream_client:
            response = stream_client.post("/chat/stream", json={"question": "cats"})
        events = _parse_sse(response.text)
        kind, data = events[0]
        assert kind == "sources"
        payload = json.loads(data)
        assert [item["chunk_index"] for item in payload["sources"]] == [0, 1]
        assert payload["sources"][0]["document_id"] == "a-doc"
        assert "b-doc" not in data

    def test_tokens_are_json_text_deltas(self, monkeypatch, users, ready_store):
        with _client(monkeypatch, users["a"], FakeStreamLLM()) as stream_client:
            response = stream_client.post("/chat/stream", json={"question": "cats"})
        events = _parse_sse(response.text)
        token_texts = [json.loads(data)["text"] for name, data in events if name == "token"]
        assert token_texts == ["hello ", "from ", "the ", "fake ", "stream"]
        assert "".join(token_texts) == "hello from the fake stream"

    def test_no_ready_documents_streams_message_without_llm(
        self, monkeypatch, users, no_ready_store
    ):
        FakeStreamLLM.calls = []
        with _client(monkeypatch, users["a"], FakeStreamLLM()) as stream_client:
            response = stream_client.post("/chat/stream", json={"question": "cats"})
        events = _parse_sse(response.text)
        names = [name for name, data in events]
        assert names == ["sources", "token", "done"]
        assert json.loads(events[0][1])["sources"] == []
        assert json.loads(events[1][1])["text"] == NO_DOCUMENTS_MESSAGE
        assert not FakeStreamLLM.calls

    def test_stream_errors_become_error_events(self, monkeypatch, users, ready_store):
        for error, expected in (
            (FakeRateLimitError("groq busy; internal details X"), RATE_LIMIT_MESSAGE),
            (FakeAuthError("bad key details Y"), AUTH_ERROR_MESSAGE),
            (RuntimeError("sneaky traceback Z"), GENERIC_ERROR_MESSAGE),
        ):
            StreamFailLLM.error_to_raise = error
            with _client(monkeypatch, users["a"], StreamFailLLM()) as stream_client:
                response = stream_client.post("/chat/stream", json={"question": "cats"})
            events = _parse_sse(response.text)
            names = [name for name, data in events]
            assert "error" in names, f"missing error event for {error!r}"
            payload = json.loads([data for name, data in events if name == "error"][0])
            assert payload["error"] == expected
            # Friendly only: the raw exception text must never reach the client.
            assert "internal details X" not in response.text
            assert "bad key details Y" not in response.text
            assert "sneaky traceback Z" not in response.text
            assert "Traceback" not in response.text


class TestFriendlyErrorMapping:
    def test_429_status(self):
        class Custom(Exception):
            status_code = 429

        assert friendly_error(Custom()) == RATE_LIMIT_MESSAGE

    def test_401_status(self):
        class Custom(Exception):
            status_code = 401

        assert friendly_error(Custom()) == AUTH_ERROR_MESSAGE

    def test_generic_message_never_leaks_details(self):
        assert friendly_error(ValueError("the JINA key is dead")) == GENERIC_ERROR_MESSAGE
