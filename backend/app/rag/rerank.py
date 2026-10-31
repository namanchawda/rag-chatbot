"""Jina-hosted reranker exposed as a small scoring client.

Calls POST https://api.jina.ai/v1/rerank with httpx. The reply maps results
back to the input documents by index. Relevance scores are raw model
outputs (roughly -1..1 but not guaranteed, not a probability): callers must
never assume a 0..1 range or a fixed threshold.

Retries with exponential backoff on 429 and 5xx, throttles requests like the
embeddings client, and surfaces rejected-model errors verbatim.
"""

import threading
import time
import logging
from typing import Any

import httpx

from ..config import get_settings
from .status import (
    SERVICE_JINA,
    STATE_EXHAUSTED,
    STATE_INVALID_KEY,
    ServiceUnavailableError,
    classify_failure,
    record_failure,
    record_success,
    usage_tokens,
)

logger = logging.getLogger(__name__)

_API_URL = "https://api.jina.ai/v1/rerank"

_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_CAP_SECONDS = 8.0

# Same conservative spacing as the embeddings client so bursts stay under
# Jina's ~100 requests/minute limit.
_MIN_REQUEST_SPACING_SECONDS = 60.0 / 110.0


class JinaRerankError(RuntimeError):
    """Raised when the Jina rerank API cannot be used as configured."""


def _parse_scores(response: httpx.Response) -> list[tuple[int, float]]:
    """Parse a reply into (index, relevance_score) pairs, returned as-is."""
    if response.status_code != 200:
        state, _message = classify_failure(SERVICE_JINA, response.status_code, response.text)
        record_failure(SERVICE_JINA, response.status_code, response.text)
        if state in (STATE_EXHAUSTED, STATE_INVALID_KEY):
            # Client-safe message; the retrieval fallback swallows it and
            # chat proceeds on vector order without reranking.
            raise ServiceUnavailableError()
        # Keep the API's exact error text: a rejected model name must be
        # visible to operators.
        raise JinaRerankError(
            f"Jina rerank API returned {response.status_code}: {response.text}"
        )
    payload = response.json()
    record_success(SERVICE_JINA, usage_tokens(payload))
    if not isinstance(payload, dict) or "results" not in payload:
        raise JinaRerankError(f"Unexpected Jina rerank response: {str(payload)[:300]}")
    parsed: list[tuple[int, float]] = []
    for item in payload.get("results", []):
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or not isinstance(score, (int, float)):
            raise JinaRerankError(
                f"Jina rerank result is missing index or relevance_score: {item}"
            )
        if index < 0:
            raise JinaRerankError(f"Jina rerank returned a negative index: {item}")
        parsed.append((index, float(score)))
    return parsed


class JinaReranker:
    """Scores candidate chunks against one question with the Jina rerank API.

    Returns (index, score) pairs where the index refers to the position of
    the text in the documents argument. Passing a prebuilt httpx Client
    (with a MockTransport) keeps tests hermetic.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
    ):
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.JINA_API_KEY
        if not self._api_key:
            raise JinaRerankError(
                "JINA_API_KEY is not configured. Set it in backend/.env before reranking."
            )
        self._model = model if model is not None else settings.JINA_RERANK_MODEL
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0))
        self._rate_lock = threading.Lock()
        self._last_request_at = 0.0

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        """Score documents against the query.

        By default all documents are scored (top_n defaults to their count)
        so callers can log every candidate and cut on min_score themselves.
        """
        if not documents:
            return []
        request_top_n = len(documents) if top_n is None else max(1, int(top_n))
        body: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "top_n": request_top_n,
            "documents": documents,  # Plain strings are accepted.
        }
        return self._request(body)

    def _request(self, body: dict[str, Any]) -> list[tuple[int, float]]:
        last_error: Exception | None = None
        last_failure_response: httpx.Response | None = None
        last_network_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            self._throttle()
            try:
                response = self._client.post(
                    _API_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
            except httpx.HTTPError as error:
                last_error = JinaRerankError(f"Jina rerank request failed: {error}")
                last_network_error = error
                response = None

            if response is None:
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 429 or 500 <= response.status_code < 600:
                last_error = JinaRerankError(
                    f"Jina rerank API returned {response.status_code}: {response.text[:300]}"
                )
                last_failure_response = response
                self._sleep_backoff(attempt, retry_after=response.headers.get("Retry-After"))
                continue
            return _parse_scores(response)

        # Retries exhausted: record the final failure only.
        if last_failure_response is not None:
            record_failure(SERVICE_JINA, last_failure_response.status_code, last_failure_response.text)
        elif last_network_error is not None:
            record_failure(SERVICE_JINA, None, f"Jina rerank request failed: {last_network_error}")
        raise JinaRerankError(
            f"Jina rerank API kept failing after {_MAX_ATTEMPTS} attempts: {last_error}"
        )

    def _throttle(self) -> None:
        with self._rate_lock:
            now = time.monotonic()
            wait = self._last_request_at + _MIN_REQUEST_SPACING_SECONDS - now
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    def _sleep_backoff(self, attempt: int, retry_after: str | None = None) -> None:
        delay = _BACKOFF_BASE_SECONDS * (2**attempt)
        delay = min(delay, _BACKOFF_CAP_SECONDS)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(delay)
