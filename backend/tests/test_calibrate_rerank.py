"""Calibration helper and summary-logic tests.

Real retrieval path with a fake vector store and a mocked Jina reranker
(httpx.MockTransport). No real Jina, Groq, or Neon calls.
"""

import httpx
import pytest

from app.rag import retrieve as retrieve_module
from app.rag.retrieve import (
    retrieve_reranked_chunks,
    retrieve_scored_candidates,
)
from scripts.calibrate_rerank import compute_summary, format_summary
from tests.test_rerank import (
    USER_A,
    USER_A_STR,
    UserFilteredStore,
    _chunk,
    enable_rerank,
    make_transport,
)

ALL_STORE_DOCS = [
    _chunk(USER_A_STR, "standard working hours text", "doc-a", 0),
    _chunk(USER_A_STR, "casual leave allowance text", "doc-b", 0),
    _chunk(USER_A_STR, "salary payday text", "doc-c", 0),
    _chunk(USER_A_STR, "probation period text", "doc-d", 0),
    _chunk("other-user", "secret other text", "doc-z", 0),
]


def _store() -> UserFilteredStore:
    return UserFilteredStore(ALL_STORE_DOCS)


def _mock_reranker(scores: list[float]):
    def handler(request: httpx.Request) -> httpx.Response:
        results = [
            {"index": index, "relevance_score": score}
            for index, score in enumerate(scores)
        ]
        return httpx.Response(
            200,
            json={"results": results},
            request=request,
        )

    return make_transport(handler)


def _top_scores(store, questions_with_scores, monkeypatch):
    """Run each question through the real helper; return top scores."""
    enable_rerank(monkeypatch)
    tops = []
    for scores in questions_with_scores:
        scored = retrieve_scored_candidates(
            "unused question", USER_A, store=store, reranker=_mock_reranker(scores)
        )
        assert all(score is not None for _, score in scored)
        tops.append(max(score for _, score in scored))
    return tops


class TestRetrieveScoredCandidates:
    def test_returns_every_candidate_with_score_no_cutoff_or_slicing(
        self, monkeypatch
    ):
        # A cutoff that would remove everything in production must not touch
        # the calibration helper.
        enable_rerank(monkeypatch, RERANK_MIN_SCORE=0.9)
        store = _store()
        scored = retrieve_scored_candidates(
            "q", USER_A, store=store, reranker=_mock_reranker([0.5, 0.2, 0.4, -0.1])
        )
        assert [score for _, score in scored] == [0.5, 0.4, 0.2, -0.1]
        assert len(scored) == 4

    def test_keeps_the_production_user_filter(self, monkeypatch):
        enable_rerank(monkeypatch)
        store = _store()
        retrieve_scored_candidates(
            "q", USER_A, store=store, reranker=_mock_reranker([0.0, 0.0, 0.0, 0.0])
        )
        assert store.filters[-1] == {"user_id": USER_A_STR}

    def test_production_path_still_applies_cutoff_and_top_n(self, monkeypatch):
        enable_rerank(monkeypatch, RERANK_MIN_SCORE=0.9, RERANK_TOP_N=2)
        store = _store()
        chunks = retrieve_reranked_chunks(
            "q", USER_A, store=store, reranker=_mock_reranker([0.5, 0.2, 0.4, -0.1])
        )
        assert chunks == []  # everything below the cutoff

        enable_rerank(monkeypatch, RERANK_MIN_SCORE=None, RERANK_TOP_N=2)
        chunks = retrieve_reranked_chunks(
            "q", USER_A, store=store, reranker=_mock_reranker([0.5, 0.2, 0.4, -0.1])
        )
        assert [chunk.document_id for chunk in chunks] == ["doc-a", "doc-c"]


class TestSummaryLogic:
    def test_clean_gap_suggests_midpoint(self, monkeypatch):
        store = _store()
        answerable_tops = _top_scores(
            store,
            [[0.4, 0.2, 0.1, 0.05], [0.3, 0.1, 0.0, -0.1]],
            monkeypatch,
        )
        unanswerable_tops = _top_scores(
            _store(),
            [[0.1, 0.05, 0.0, -0.3], [-0.2, -0.4, -0.5, -0.6]],
            monkeypatch,
        )
        summary = compute_summary(answerable_tops, unanswerable_tops)
        assert summary["has_gap"] is True
        assert summary["lowest_answerable"] == pytest.approx(0.3)
        assert summary["highest_unanswerable"] == pytest.approx(0.1)
        assert summary["midpoint"] == pytest.approx(0.2)
        text = format_summary(summary)
        assert "Clean gap. Suggested RERANK_MIN_SCORE = 0.2000" in text
        assert "gap width 0.2000" in text

    def test_overlap_reports_both_numbers(self, monkeypatch):
        answerable_tops = _top_scores(
            _store(),
            [[0.25, 0.1, 0.05, 0.0], [0.1, 0.05, 0.0, -0.1]],
            monkeypatch,
        )
        unanswerable_tops = _top_scores(
            _store(),
            [[0.44, 0.2, 0.1, -0.1], [0.4, 0.3, 0.1, 0.0]],
            monkeypatch,
        )
        summary = compute_summary(answerable_tops, unanswerable_tops)
        assert summary["has_gap"] is False
        text = format_summary(summary)
        assert (
            "Groups overlap. Choose a cutoff below the lowest answerable score"
            in text
        )
        assert "accept some false positives" in text
        assert f"({summary['lowest_answerable']:.4f})" in text
        assert f"{summary['highest_unanswerable']:.4f}" in text
