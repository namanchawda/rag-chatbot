import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from .config import get_settings

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user_id: uuid.UUID) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=get_settings().JWT_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, get_settings().JWT_SECRET, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(token, get_settings().JWT_SECRET, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None


def is_admin_email(email: str) -> bool:
    """True when the email is on ADMIN_EMAILS (case-insensitive).

    An empty list means nobody is an admin. The admin list itself is never
    exposed anywhere; this returns only a yes/no computed server-side.
    """
    allowed = get_settings().ADMIN_EMAILS or ""
    admins = {entry.strip().lower() for entry in allowed.split(",") if entry.strip()}
    if not admins:
        return False
    return (email or "").strip().lower() in admins
