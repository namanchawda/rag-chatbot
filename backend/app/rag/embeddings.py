"""Jina-hosted embeddings exposed as a LangChain Embeddings implementation.

Calls POST https://api.jina.ai/v1/embeddings with httpx. Batches at most 32
texts per request, throttles to roughly 100 requests/minute and 100K tokens
per minute, and retries with exponential backoff on 429 and 5xx responses.
"""

import threading
import time

import httpx
from langchain_core.embeddings import Embeddings

from ..config import get_settings

_API_URL = "https://api.jina.ai/v1/embeddings"
_MAX_BATCH = 32
_PASSAGE_TASK = "retrieval.passage"
_QUERY_TASK = "retrieval.query"

_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_CAP_SECONDS = 8.0

# Slightly conservative spacing so bursts stay under ~100 requests/minute and
# the ~100K tokens/minute limit even with full 32-text batches.
_MIN_REQUEST_SPACING_SECONDS = 60.0 / 110.0


class JinaEmbeddingError(RuntimeError):
    """Raised when the Jina embeddings API cannot be used as configured."""


class JinaEmbeddings(Embeddings):
    """Embeddings backed by the Jina embeddings API.

    Returns vectors with the dimension taken from EMBEDDING_DIMENSION, so the
    pgvector column shape is always consistent with the configured model
    settings.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None, dimensions: int | None = None):
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.JINA_API_KEY
        if not self._api_key:
            raise JinaEmbeddingError(
                "JINA_API_KEY is not configured. Set it in backend/.env before ingesting documents."
            )
        self._model = model if model is not None else settings.JINA_EMBEDDING_MODEL
        self._dimensions = dimensions if dimensions is not None else settings.EMBEDDING_DIMENSION
        self._client = httpx.Client(timeout=httpx.Timeout(60.0))
        self._rate_lock = threading.Lock()
        self._last_request_at = 0.0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH):
            batch = texts[start : start + _MAX_BATCH]
            vectors.extend(self._embed_batch(batch, task=_PASSAGE_TASK))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], task=_QUERY_TASK)[0]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed_batch(self, texts: list[str], *, task: str) -> list[list[float]]:
        if not texts:
            return []
        data = self._request(texts, task=task)
        results = data.get("data", [])
        # Re-order results by the index Jina attaches to each result, so output
        # order always matches input order.
        results.sort(key=lambda item: item.get("index", 0))
        embeddings = []
        for item in results:
            embedding = item.get("embedding")
            if embedding is None:
                raise JinaEmbeddingError(
                    f"Jina embedding response is missing data[i].embedding: {item}"
                )
            embeddings.append(embedding)
        if len(embeddings) != len(texts):
            raise JinaEmbeddingError(
                f"Jina returned {len(embeddings)} embeddings for {len(texts)} input texts."
            )
        for embedding in embeddings:
            if len(embedding) != self._dimensions:
                raise JinaEmbeddingError(
                    f"Jina returned a vector of size {len(embedding)} but EMBEDDING_DIMENSION is "
                    f"{self._dimensions}. Ensure the configured model matches EMBEDDING_DIMENSION."
                )
        return embeddings

    def _request(self, batch: list[str], *, task: str) -> dict:
        body = {
            "model": self._model,
            "task": task,
            "normalized": True,
            "dimensions": self._dimensions,
            "input": [{"text": text} for text in batch],
        }

        last_error: Exception | None = None
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
                last_error = JinaEmbeddingError(f"Jina embedding request failed: {error}")
                response = None

            if response is None:
                self._sleep_backoff(attempt)
                continue

            if response.status_code in (429, *range(500, 600)):
                last_error = JinaEmbeddingError(
                    f"Jina embeddings API returned {response.status_code}: {response.text}"
                )
                self._sleep_backoff(attempt, retry_after=response.headers.get("Retry-After"))
                continue
            return self._parse(response)

        raise JinaEmbeddingError(f"Jina embeddings API kept failing after {_MAX_ATTEMPTS} attempts: {last_error}")

    def _parse(self, response: httpx.Response) -> dict:
        # Any non-retryable error is surfaced verbatim; in particular a rejected
        # task value must bubble up exactly as the API reported it.
        if response.status_code != 200:
            raise JinaEmbeddingError(
                f"Jina embeddings API returned {response.status_code}: {response.text}"
            )
        payload = response.json()
        parse_error = payload.get("error") if isinstance(payload, dict) else None
        if parse_error:
            raise JinaEmbeddingError(f"Jina embeddings API error: {parse_error}")
        if "data" not in payload:
            raise JinaEmbeddingError(f"Unexpected Jina embeddings response: {payload}")
        return payload

    def _throttle(self) -> None:
        # Serialise requests and keep a floor between starts so sustained
        # ingestion stays under Jina's 100 requests/minute and 100K tokens/minute
        # limits even when every batch fills up completely.
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
