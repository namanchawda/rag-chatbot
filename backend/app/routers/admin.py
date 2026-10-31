"""Admin-only service status endpoints.

Bearer token required; anyone not on ADMIN_EMAILS gets a plain 404 so the
endpoints look nonexistent. Responses carry states, timestamps and usage
counts only - never API keys or key fragments, and the admin list itself is
never exposed.

GET    /admin/status           current payload for jina and groq
POST   /admin/status/check     ONE minimal Jina probe, recorded, then the
                               payload (cooldown STATUS_CHECK_COOLDOWN_SECONDS,
                               never calls Groq)
POST   /admin/status/reset-usage  set Jina usage_reset_at to now (after
                               replacing the key), then the payload
"""

import math
import time

from fastapi import APIRouter, Depends, HTTPException, status

from ..config import get_settings
from ..deps import get_current_user
from ..models import User
from ..rag.status import (
    SERVICE_JINA,
    reset_usage,
    run_jina_status_check,
    status_payload,
)
from ..security import is_admin_email

router = APIRouter(prefix="/admin", tags=["admin"])

_last_check_monotonic: float | None = None


def reset_check_cooldown() -> None:
    """Clear the in-process cooldown marker (used by tests)."""
    global _last_check_monotonic
    _last_check_monotonic = None


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if not is_admin_email(current_user.email):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return current_user


@router.get("/status", dependencies=[Depends(require_admin)])
def get_admin_status() -> dict:
    return status_payload()


@router.post("/status/check", dependencies=[Depends(require_admin)])
def check_admin_status() -> dict:
    global _last_check_monotonic
    cooldown = get_settings().STATUS_CHECK_COOLDOWN_SECONDS
    if _last_check_monotonic is not None:
        remaining = cooldown - (time.monotonic() - _last_check_monotonic)
        if remaining > 0:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Status check cooldown is active. Please wait and try again.",
                headers={"Retry-After": str(int(math.ceil(remaining)))},
            )
    _last_check_monotonic = time.monotonic()
    run_jina_status_check()
    return status_payload()


@router.post("/status/reset-usage", dependencies=[Depends(require_admin)])
def reset_admin_usage() -> dict:
    reset_usage(SERVICE_JINA)
    return status_payload()
