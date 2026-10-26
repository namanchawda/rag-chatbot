"""Rerank tests. Everything is fake: no real Jina, Groq, or Neon calls.

The JinaReranker client is driven over httpx.MockTransport; the retrieval
stage uses the same UserFilteredStore contract as the chat endpoint tests,
and the LLM is a recording fake.
"""

import json
import logging
import uuid
from typing import ClassVar

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.config import get_settings
from app.deps import get_current_user
from app.main import create_app
from app.models import Document as DocumentRow
from app.models import User
from app.models import DocumentStatus
from app.rag import retrieve as retrieve_module
import app.rag.rerank as rerank_module
from app.rag.chat import NO_RESULTS_MESSAGE
from app.rag.rerank import JinaRerankError, JinaReranker
from app.rag.retrieve import retrieve_reranked_chunks
from app.routers import chat as chat_router

USER_A = uuid.UUID(int=1)
USER_A_STR = str(USER_A)


class FakeChatLLM(BaseChatModel):
    reply: str = "fake answer"
    calls: ClassVar[list] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    @property
    def _llm_type(self) -> str:
        return "fake"


class FakeStreamLLM(BaseChatModel):
    calls: ClassVar[list] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("non-streaming generation should not be used here")

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        yield ChatGenerationChunk(message=AIMessageChunk(content="token "))

    @property
    def _llm_type(self) -> str:
        return "fake-stream"


class ScriptedReranker:
    """Returns prepared (index, score) results; records what it saw."""

    def __init__(self, results=None, error: Exception | None = None):
        self.results = results if results is not None else []
        self.error = error
        self.queries: list[str] = []
        self.document_lists: list[list[str]] = []

    def rerank(self, query, documents, *, top_n=None):
        self.queries.append(query)
        self.document_lists.append(list(documents))
        if self.error is not None:
            raise self.error
        return list(self.results)


class UserFilteredStore:
    """Fake PGVector: only hands back docs matching the filter's user_id."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.filters: list[dict] = []

    def similarity_search(self, query, k=4, filter=None, **kwargs):
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


USER_A_TEXTS = [
    _chunk(USER_A_STR, "cats purr softly", "doc-a", 0),
    _chunk(USER_A_STR, "engines need oil", "doc-b", 0),
    _chunk(USER_A_STR, "cats groom daily", "doc-a", 1),
    _chunk(USER_A_STR, "cats sleep lots", "doc-a", 2),
]
OTHER_USER_TEXT = _chunk("other-user", "secret about b", "doc-c", 0)
ALL_CANDIDATES = USER_A_TEXTS + [OTHER_USER_TEXT]


def make_transport(handler) -> JinaReranker:
    return JinaReranker(
        api_key="test-key",
        client=httpx.Client(
            transport=httpx.MockTransport(handler),
            timeout=httpx.Timeout(0.1),
        ),
    )


def enable_rerank(monkeypatch, **overrides) -> None:
    settings = get_settings().model_copy(
        update={"RERANK_ENABLED": True, **overrides}
    )
    monkeypatch.setattr(retrieve_module, "get_settings", lambda: settings)


class TestRerankClient:
    def test_request_shape_and_header(self):
        request_box: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            request_box["body"] = json.loads(request.read().decode("utf-8"))
            request_box["auth"] = request.headers["Authorization"]
            return httpx.Response(
                200,
                content=json.dumps(
                    {
                        "results": [
                            {"index": 2, "relevance_score": 0.9},
                            {"index": 0, "relevance_score": 0.4},
                        ]
                    }
                ).encode(),
                request=httpx.Request("POST", "https://api.jina.ai/v1/rerank"),
            )

        scored = make_transport(handler).rerank(
            "cats?", [chunk.page_content for chunk in ALL_CANDIDATES]
        )

        assert scored == [(2, 0.9), (0, 0.4)]
        body = request_box["body"]
        assert body["model"] == "jina-reranker-v3.5"
        assert body["query"] == "cats?"
        assert body["top_n"] == 5  # scores every candidate by default
        assert body["documents"] == [
            "cats purr softly",
            "engines need oil",
            "cats groom daily",
            "cats sleep lots",
            "secret about b",
        ]
        assert request_box["auth"] == "Bearer test-key"

    def test_explicit_top_n(self):
        request_box: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            request_box["body"] = json.loads(request.read().decode("utf-8"))
            return httpx.Response(
                200,
                content=b'{"results": []}',
                request=httpx.Request("POST", "https://api.jina.ai/v1/rerank"),
            )

        make_transport(handler).rerank("q", ["a", "b", "c"], top_n=2)
        assert request_box["body"]["top_n"] == 2

    def test_rejected_model_surfaces_exact_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                content=b'{"message": "Model not found: jina-reranker-v9.9"}',
                request=httpx.Request("POST", "https://api.jina.ai/v1/rerank"),
            )

        with pytest.raises(JinaRerankError) as error:
            make_transport(handler).rerank("q", ["a"])
        assert "Model not found: jina-reranker-v9.9" in str(error.value)
        assert "400" in str(error.value)

    def test_retries_5xx_then_succeeds(self):
        state = {"calls": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["calls"] += 1
            if state["calls"] == 1:
                return httpx.Response(
                    500,
                    content=b"boom",
                    headers={"Retry-After": "0"},
                    request=httpx.Request("POST", "https://api.jina.ai/v1/rerank"),
                )
            return httpx.Response(
                200,
                content=b'{"results": [{"index": 0, "relevance_score": 0.1}]}',
                request=httpx.Request("POST", "https://api.jina.ai/v1/rerank"),
            )

        assert make_transport(handler).rerank("q", ["a"]) == [(0, 0.1)]
        assert state["calls"] == 2

    def test_empty_documents_never_calls_api(self):
        marker = {"called": False}

        def handler(request: httpx.Request) -> httpx.Response:
            marker["called"] = True
            return httpx.Response(
                200,
                content=b'{"results": []}',
                request=httpx.Request("POST", "https://api.jina.ai/v1/rerank"),
            )

        make_transport(handler).rerank("q", [])
        assert marker["called"] is False

    def test_missing_api_key_raises_clear_error(self):
        with pytest.raises(JinaRerankError, match="JINA_API_KEY"):
            JinaReranker(api_key="", model="m")


class TestRerankedRetrieval:
    def test_maps_by_index_even_out_of_order(self, monkeypatch):
        store = UserFilteredStore(ALL_CANDIDATES)
        enable_rerank(monkeypatch)
        # Relevance order, unrelated to candidate order: index 2 first.
        reranker = ScriptedReranker(
            results=[(2, 0.9), (0, 0.15), (3, -0.17), (1, -0.05)]
        )

        chunks = retrieve_reranked_chunks(
            "cats", USER_A, store=store, reranker=reranker
        )
        assert store.filters[-1] == {"user_id": USER_A_STR}
        # Mapped by index, best score first: 0.9 -> 0.15 -> -0.05 -> -0.17.
        assert [chunk.text for chunk in chunks] == [
            "cats groom daily",     # index 2
            "cats purr softly",     # index 0
            "engines need oil",     # index 1
            "cats sleep lots",      # index 3
        ]
        # And the reranker only ever saw the caller's own chunks.
        assert reranker.document_lists[0] == [
            "cats purr softly",
            "engines need oil",
            "cats groom daily",
            "cats sleep lots",
        ]

    def test_cutoff_drops_low_scores(self, monkeypatch):
        store = UserFilteredStore(ALL_CANDIDATES)
        enable_rerank(monkeypatch, RERANK_MIN_SCORE=0.0)
        reranker = ScriptedReranker(
            results=[(2, 0.9), (0, 0.15), (3, -0.17), (1, -0.05)]
        )

        chunks = retrieve_reranked_chunks(
            "cats", USER_A, store=store, reranker=reranker
        )
        # Cutoff drops everything below RERANK_MIN_SCORE (index 3 at -0.17
        # and index 1 at -0.05), the rest stays ranked.
        assert [chunk.text for chunk in chunks] == [
            "cats groom daily",     # index 2 @ 0.9
            "cats purr softly",     # index 0 @ 0.15
        ]

    def test_top_n_limits_kept_chunks(self, monkeypatch):
        store = UserFilteredStore(ALL_CANDIDATES)
        enable_rerank(monkeypatch, RERANK_TOP_N=2)
        reranker = ScriptedReranker(
            results=[(2, 0.9), (0, 0.15), (3, 0.14), (1, 0.12)]
        )

        chunks = retrieve_reranked_chunks(
            "cats", USER_A, store=store, reranker=reranker
        )
        assert len(chunks) == 2
        assert chunks[0].text == "cats groom daily"
        assert chunks[1].text == "cats purr softly"

    def test_reranker_failure_falls_back_to_vector_order(self, monkeypatch):
        store = UserFilteredStore(ALL_CANDIDATES)
        enable_rerank(monkeypatch, RERANK_TOP_N=2)
        reranker = ScriptedReranker(error=RuntimeError("timeout"))

        chunks = retrieve_reranked_chunks(
            "cats", USER_A, store=store, reranker=reranker
        )
        # First RERANK_TOP_N vector results, no exception propagated.
        assert [chunk.text for chunk in chunks] == [
            "cats purr softly",
            "engines need oil",
        ]

    def test_user_filter_still_applied_when_reranking(self, monkeypatch):
        store = UserFilteredStore(ALL_CANDIDATES)
        enable_rerank(monkeypatch)
        reranker = ScriptedReranker()

        retrieve_reranked_chunks("cats", USER_A, store=store, reranker=reranker)
        assert store.filters[0] == {"user_id": USER_A_STR}
        # The other user's chunk was never fetched, so it never reached the
        # reranker and can never come back as a used chunk.
        assert len(reranker.document_lists[0]) == 4
        assert "secret about b" not in reranker.document_lists[0]

    def test_ranked_output_prohibits_other_users_chunk(self, monkeypatch):
        store = UserFilteredStore(ALL_CANDIDATES)
        enable_rerank(monkeypatch)
        reranker = ScriptedReranker(results=[(0, 0.5), (1, 0.4)])

        chunks = retrieve_reranked_chunks(
            "cats", USER_A, store=store, reranker=reranker
        )
        assert all("secret about b" != chunk.text for chunk in chunks)

    def test_disabled_uses_first_top_n_vector_order(self, monkeypatch):
        settings = get_settings().model_copy(
            update={"RERANK_ENABLED": False, "RERANK_TOP_N": 2}
        )
        monkeypatch.setattr(retrieve_module, "get_settings", lambda: settings)
        store = UserFilteredStore(ALL_CANDIDATES)

        chunks = retrieve_reranked_chunks("cats", USER_A, store=store)
        assert [chunk.text for chunk in chunks] == [
            "cats purr softly",
            "engines need oil",
        ]

    def test_empty_rerank_results_return_empty_without_llm(self, monkeypatch):
        store = UserFilteredStore(ALL_CANDIDATES)
        enable_rerank(monkeypatch)
        reranker = ScriptedReranker(results=[])

        assert (
            retrieve_reranked_chunks("cats", USER_A, store=store, reranker=reranker)
            == []
        )
        assert reranker.queries == ["cats"]  # candidates are the 4 user-A chunks
        assert len(reranker.document_lists[0]) == 4
        assert "secret about b" not in reranker.document_lists[0]

    def test_all_below_cutoff_returns_empty(self, monkeypatch):
        store = UserFilteredStore(ALL_CANDIDATES)
        enable_rerank(monkeypatch, RERANK_MIN_SCORE=0.0)
        reranker = ScriptedReranker(
            results=[(0, -0.10), (1, -0.20), (2, -0.30), (3, -0.40)]
        )

        assert (
            retrieve_reranked_chunks("cats", USER_A, store=store, reranker=reranker)
            == []
        )


class TestScoreLogging:
    def test_logs_truncated_question_and_index_score_pairs(
        self, monkeypatch, caplog
    ):
        store = UserFilteredStore(ALL_CANDIDATES)
        enable_rerank(monkeypatch, RERANK_LOG_SCORES=True)
        reranker = ScriptedReranker(
            results=[(2, 0.14), (0, 0.11), (1, 0.03), (3, -0.17)]
        )

        with caplog.at_level(logging.INFO, logger="app.rag.retrieve"):
            retrieve_reranked_chunks(
                "cats", USER_A, store=store, reranker=reranker
            )

        records = [
            record
            for record in caplog.records
            if "Rerank scores" in record.getMessage()
        ]
        assert len(records) == 1
        message = records[0].getMessage()
        assert "question='cats'" in message
        for pair in ("2:0.1400", "0:0.1100", "1:0.0300", "3:-0.1700"):
            assert pair in message
        # No chunk text, no document ids in the log line.
        assert "cats purr softly" not in message
        assert "secret about b" not in message
        assert "doc-a" not in message

    def test_scores_not_logged_when_disabled(self, monkeypatch, caplog):
        store = UserFilteredStore(ALL_CANDIDATES)
        enable_rerank(monkeypatch, RERANK_LOG_SCORES=False)
        reranker = ScriptedReranker(results=[(2, 0.9)])

        with caplog.at_level(logging.INFO, logger="app.rag.retrieve"):
            retrieve_reranked_chunks(
                "cats", USER_A, store=store, reranker=reranker
            )
        assert not [
            record
            for record in caplog.records
            if "Rerank scores" in record.getMessage()
        ]

    def test_long_question_is_truncated_to_60_chars(self, monkeypatch, caplog):
        store = UserFilteredStore(ALL_CANDIDATES)
        enable_rerank(monkeypatch, RERANK_LOG_SCORES=True)
        reranker = ScriptedReranker(results=[(0, 0.5)])
        question = "q" * 100

        with caplog.at_level(logging.INFO, logger="app.rag.retrieve"):
            retrieve_reranked_chunks(
                question, USER_A, store=store, reranker=reranker
            )
        message = [
            record
            for record in caplog.records
            if "Rerank scores" in record.getMessage()
        ][0].getMessage()
        logged_question = message.split("question='")[1].split("'")[0]
        assert len(logged_question) == 60
        assert logged_question == "q" * 60


class TestChatEndpointsWithRerank:
    def _make_ready_document(self, db_session, user_id):
        row = DocumentRow(
            id=uuid.uuid4(),
            user_id=user_id,
            filename="doc-a.txt",
            status=DocumentStatus.READY,
            chunk_count=3,
        )
        db_session.add(row)
        db_session.commit()
        return row

    def _setup_endpoint(self, monkeypatch, db_session, llm):
        user = User(id=USER_A, email="rerank@example.com", password_hash="x")
        db_session.add(user)
        db_session.commit()
        self._make_ready_document(db_session, USER_A)

        store = UserFilteredStore(ALL_CANDIDATES)
        monkeypatch.setattr(retrieve_module, "build_vectorstore", lambda: store)
        monkeypatch.setattr(chat_router, "build_llm", lambda: llm)

        app = create_app()
        app.dependency_overrides[get_current_user] = lambda: user
        return TestClient(app)

    def test_endpoint_sources_follow_reranked_order(
        self, monkeypatch, db_session
    ):
        reranker = ScriptedReranker(
            results=[(2, 0.9), (1, 0.4), (0, 0.15)]
        )
        monkeypatch.setattr(
            rerank_module, "JinaReranker", lambda *args, **kwargs: reranker
        )
        enable_rerank(monkeypatch)
        FakeChatLLM.calls = []
        test_client = self._setup_endpoint(
            monkeypatch, db_session, FakeChatLLM(reply="ok")
        )

        response = test_client.post("/chat", json={"question": "cats"})
        assert response.status_code == 200
        # Sources are exactly the chunks used for the answer, in reranked
        # order (candidates 2, 1, 0 — not the vector order 0, 1, 2).
        sources = response.json()["sources"]
        assert [(s["chunk_index"], s["document_id"]) for s in sources] == [
            (1, "doc-a"),  # candidate index 2
            (0, "doc-b"),  # candidate index 1
            (0, "doc-a"),  # candidate index 0
        ]
        assert FakeChatLLM.calls, "answer expected for a good candidate"

    def test_all_below_cutoff_skips_llm(self, monkeypatch, db_session):
        reranker = ScriptedReranker(
            results=[(0, -0.10), (1, -0.20), (2, -0.30)]
        )
        monkeypatch.setattr(
            rerank_module, "JinaReranker", lambda *args, **kwargs: reranker
        )
        enable_rerank(monkeypatch, RERANK_MIN_SCORE=0.0)
        FakeChatLLM.calls = []
        test_client = self._setup_endpoint(
            monkeypatch, db_session, FakeChatLLM(reply="never")
        )

        response = test_client.post("/chat", json={"question": "cats"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == NO_RESULTS_MESSAGE
        assert body["sources"] == []
        assert FakeChatLLM.calls == []

    def test_reranker_failure_falls_back_and_stream_order_unchanged(
        self, monkeypatch, db_session
    ):
        reranker = ScriptedReranker(error=RuntimeError("503"))
        monkeypatch.setattr(
            rerank_module, "JinaReranker", lambda *args, **kwargs: reranker
        )
        enable_rerank(monkeypatch)
        FakeStreamLLM.calls = []
        test_client = self._setup_endpoint(
            monkeypatch, db_session, FakeStreamLLM()
        )

        response = test_client.post("/chat/stream", json={"question": "cats"})
        assert response.status_code == 200
        text = response.text
        names = [
            line.removeprefix("event: ").strip()
            for line in text.splitlines()
            if line.startswith("event: ")
        ]
        # SSE order unchanged: sources first, tokens, done last — and no
        # error event, since the reranker failure is a silent fallback.
        assert names[0] == "sources"
        assert names[-1] == "done"
        assert set(names[1:-1]) == {"token"}
        assert "error" not in names
        assert FakeStreamLLM.calls
