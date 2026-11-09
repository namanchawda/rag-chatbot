import hmac
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import get_current_user
from ..models import AuthCode, User, _utcnow
from ..rate_limit import allow, client_ip
from ..schemas import (
    AuthResponse,
    EmailCodeRequest,
    LoginRequest,
    ResetPasswordRequest,
    SignupRequest,
    UserResponse,
    VerifyEmailRequest,
)
from ..security import (
    create_access_token,
    hash_password,
    is_admin_email,
    verify_password,
)
from ..services.email import EmailSendError, send_email

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)

GENERIC_LOGIN_ERROR = "Invalid email or password"
INVALID_CODE_ERROR = "Invalid or expired code"
NEUTRAL_EMAIL_MESSAGE = "If an account matches that email, instructions will be sent shortly."


def _user_response(user: User) -> UserResponse:
    """Build the user payload with is_admin computed server-side."""
    return UserResponse(
        id=user.id,
        email=user.email,
        created_at=user.created_at,
        is_admin=is_admin_email(user.email),
    )


def _auth_response(user: User) -> AuthResponse:
    return AuthResponse(
        access_token=create_access_token(user.id),
        token_type="bearer",
        user=_user_response(user),
    )


def _restrict_attempts(request: Request) -> None:
    allowed, retry_after = allow(client_ip(request))
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts. Please wait a minute and try again.",
            headers={"Retry-After": str(retry_after)},
        )


def _code_digest(user_id: uuid.UUID, purpose: str, code: str) -> str:
    message = f"{user_id}:{purpose}:{code}".encode("utf-8")
    return hmac.digest(get_settings().JWT_SECRET.encode("utf-8"), message, "sha256").hex()


def _utc_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _can_send_code(db: Session, user: User, now: datetime) -> bool:
    settings = get_settings()
    latest = db.scalar(
        select(AuthCode)
        .where(AuthCode.user_id == user.id)
        .order_by(AuthCode.created_at.desc())
        .limit(1)
    )
    if latest is not None:
        cooldown = timedelta(seconds=max(0, settings.EMAIL_RESEND_COOLDOWN_SECONDS))
        if _utc_datetime(latest.created_at) > now - cooldown:
            return False
    hourly_count = db.scalar(
        select(func.count())
        .select_from(AuthCode)
        .where(AuthCode.user_id == user.id)
        .where(AuthCode.created_at > now - timedelta(hours=1))
    )
    return int(hourly_count or 0) < 5


def _issue_code(db: Session, user: User, purpose: str) -> bool:
    db.scalar(select(User.id).where(User.id == user.id).with_for_update())
    now = _utcnow()
    if not _can_send_code(db, user, now):
        return False

    code = f"{secrets.randbelow(1_000_000):06d}"
    db.execute(
        update(AuthCode)
        .where(AuthCode.user_id == user.id, AuthCode.purpose == purpose)
        .where(AuthCode.used_at.is_(None))
        .values(used_at=now)
    )
    auth_code = AuthCode(
        user_id=user.id,
        purpose=purpose,
        code_hash=_code_digest(user.id, purpose, code),
        expires_at=now + timedelta(minutes=get_settings().CODE_TTL_MINUTES),
        created_at=now,
        attempts=0,
    )
    db.add(auth_code)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise EmailSendError() from None

    duration = get_settings().CODE_TTL_MINUTES
    verb = "verify your email" if purpose == "verify" else "reset your password"
    subject = (
        f"{code} is your Naman Labs verification code"
        if purpose == "verify"
        else f"{code} is your Naman Labs password reset code"
    )
    text = (
        f"Use this code to {verb} for Naman Labs: {code}\n"
        f"It is valid for {duration} minutes. "
        "If you didn’t request this, you can ignore this email."
    )
    html = (
        f"<p>Use this code to {verb} for Naman Labs:</p><p><strong>{code}</strong></p>"
        f"<p>It is valid for {duration} minutes. "
        "If you didn’t request this, you can ignore this email.</p>"
    )
    send_email(user.email, subject, text, html)
    return True


def _latest_unused_code(db: Session, user: User, purpose: str) -> AuthCode | None:
    return db.scalar(
        select(AuthCode)
        .where(
            AuthCode.user_id == user.id,
            AuthCode.purpose == purpose,
            AuthCode.used_at.is_(None),
        )
        .order_by(AuthCode.created_at.desc())
        .limit(1)
        .with_for_update()
    )


def _check_code(db: Session, user: User, purpose: str, submitted: str) -> AuthCode | None:
    row = _latest_unused_code(db, user, purpose)
    if row is None:
        # Keep the no-code path doing a constant-time digest comparison too.
        hmac.compare_digest(_code_digest(user.id, purpose, submitted), "0" * 64)
        return None

    matches = hmac.compare_digest(
        row.code_hash,
        _code_digest(user.id, purpose, submitted),
    )
    now = _utcnow()
    expired = _utc_datetime(row.expires_at) <= now
    exhausted = row.attempts >= max(1, get_settings().CODE_MAX_ATTEMPTS)
    if not matches or expired or exhausted:
        if not exhausted:
            db.execute(
                update(AuthCode)
                .where(
                    AuthCode.id == row.id,
                    AuthCode.attempts < max(1, get_settings().CODE_MAX_ATTEMPTS),
                )
                .values(attempts=AuthCode.attempts + 1)
            )
            db.commit()
        return None
    return row


def _neutral_email_response() -> dict[str, str]:
    return {"message": NEUTRAL_EMAIL_MESSAGE}


def _send_neutrally(db: Session, user: User | None, purpose: str) -> None:
    if user is None:
        return
    if purpose == "verify" and user.email_verified:
        return
    try:
        _issue_code(db, user, purpose)
    except EmailSendError as error:
        logger.warning("Account email request could not be delivered (%s)", type(error).__name__)


@router.post(
    "/signup",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_restrict_attempts)],
)
def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> dict[str, str]:
    if not get_settings().SIGNUPS_ENABLED:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="New signups are currently closed.")
    existing = db.scalar(select(User).where(User.email == payload.email))
    if existing is not None:
        if existing.email_verified:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
        try:
            _issue_code(db, existing, "verify")
        except EmailSendError as error:
            raise HTTPException(status_code=503, detail=str(error)) from None
        return {"status": "verification_required", "email": existing.email}

    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        email_verified=False,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    try:
        _issue_code(db, user, "verify")
    except EmailSendError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    return {"status": "verification_required", "email": user.email}


@router.post(
    "/verify-email",
    dependencies=[Depends(_restrict_attempts)],
)
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)) -> AuthResponse:
    user = db.scalar(select(User).where(User.email == payload.email))
    if user is None:
        hmac.compare_digest("0" * 64, _code_digest(uuid.UUID(int=0), "verify", payload.code))
        raise HTTPException(status_code=400, detail=INVALID_CODE_ERROR)
    auth_code = _check_code(db, user, "verify", payload.code)
    if auth_code is None:
        raise HTTPException(status_code=400, detail=INVALID_CODE_ERROR)
    now = _utcnow()
    claimed = db.execute(
        update(AuthCode)
        .where(AuthCode.id == auth_code.id, AuthCode.used_at.is_(None))
        .values(used_at=now)
    )
    if claimed.rowcount != 1:
        db.rollback()
        raise HTTPException(status_code=400, detail=INVALID_CODE_ERROR)
    user.email_verified = True
    db.commit()
    return _auth_response(user)


@router.post(
    "/resend-verification",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_restrict_attempts)],
)
def resend_verification(payload: EmailCodeRequest, db: Session = Depends(get_db)) -> dict[str, str]:
    user = db.scalar(select(User).where(User.email == payload.email))
    _send_neutrally(db, user, "verify")
    return _neutral_email_response()


@router.post(
    "/login",
    dependencies=[Depends(_restrict_attempts)],
    response_model=None,
)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse | JSONResponse:
    user = db.scalar(select(User).where(User.email == payload.email))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=GENERIC_LOGIN_ERROR)
    if not user.email_verified:
        try:
            _issue_code(db, user, "verify")
        except EmailSendError as error:
            raise HTTPException(status_code=503, detail=str(error)) from None
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "detail": "Please verify your email to continue.",
                "code": "email_not_verified",
                "email": user.email,
            },
        )
    return _auth_response(user)


@router.post(
    "/forgot-password",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_restrict_attempts)],
)
def forgot_password(payload: EmailCodeRequest, db: Session = Depends(get_db)) -> dict[str, str]:
    user = db.scalar(select(User).where(User.email == payload.email))
    _send_neutrally(db, user, "reset")
    return _neutral_email_response()


@router.post(
    "/reset-password",
    dependencies=[Depends(_restrict_attempts)],
)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)) -> dict[str, str]:
    user = db.scalar(select(User).where(User.email == payload.email))
    if user is None:
        hmac.compare_digest("0" * 64, _code_digest(uuid.UUID(int=0), "reset", payload.code))
        raise HTTPException(status_code=400, detail=INVALID_CODE_ERROR)
    auth_code = _check_code(db, user, "reset", payload.code)
    if auth_code is None:
        raise HTTPException(status_code=400, detail=INVALID_CODE_ERROR)
    now = _utcnow()
    claimed = db.execute(
        update(AuthCode)
        .where(AuthCode.id == auth_code.id, AuthCode.used_at.is_(None))
        .values(used_at=now)
    )
    if claimed.rowcount != 1:
        db.rollback()
        raise HTTPException(status_code=400, detail=INVALID_CODE_ERROR)
    user.password_hash = hash_password(payload.new_password)
    user.password_changed_at = now
    user.email_verified = True
    db.execute(
        update(AuthCode)
        .where(AuthCode.user_id == user.id, AuthCode.used_at.is_(None))
        .values(used_at=now)
    )
    db.commit()
    return {"status": "password_reset"}


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return _user_response(current_user)
