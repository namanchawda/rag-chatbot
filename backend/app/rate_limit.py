"""In-memory sliding-window rate limiter for auth attempts.

Per process: counters reset when the app restarts, which is fine for a
single-instance deployment; a shared store would be needed for replicas.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque

from fastapi import Request

from .config import get_settings

_WINDOW_SECONDS = 60.0

_hits: dict[str, deque[float]] = {}
_lock = threading.Lock()
_last_cleanup = time.monotonic()


def monotonic_now() -> float:
    """Clock seam so tests can pass a fixed time instead of sleeping."""
    return time.monotonic()


def reset() -> None:
    with _lock:
        _hits.clear()
    global _last_cleanup
    _last_cleanup = monotonic_now()


def _prune_empty_keys(now: float) -> None:
    """Occasionally drop keys whose whole window has passed, to bound memory."""
    global _last_cleanup
    if now - _last_cleanup < _WINDOW_SECONDS:
        return
    stale = [key for key, hits in _hits.items() if not hits or now - hits[-1] >= _WINDOW_SECONDS]
    for key in stale:
        _hits.pop(key, None)
    _last_cleanup = now


def allow(key: str, now: float | None = None) -> tuple[bool, int]:
    """Record one attempt for key. Returns (allowed, retry_after_seconds)."""
    settings = get_settings()
    limit = max(1, settings.AUTH_RATE_LIMIT_PER_MINUTE)
    current = monotonic_now() if now is None else float(now)
    with _lock:
        _prune_empty_keys(current)
        hits = _hits.setdefault(key, deque())
        while hits and current - hits[0] >= _WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= limit:
            retry_after = max(1, math.ceil(_WINDOW_SECONDS - (current - hits[0])))
            return False, retry_after
        hits.append(current)
    return True, 0


def client_ip(request: Request) -> str:
    """Socket peer address by default. With TRUST_PROXY only, the rightmost
    X-Forwarded-For entry is used: that is the value a trusted platform proxy
    appended, while entries to its left can be spoofed by the client."""
    if get_settings().TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            entries = [part.strip() for part in forwarded.split(",") if part.strip()]
            if entries:
                return entries[-1]
    client = request.client
    return client.host if client is not None else "unknown"
