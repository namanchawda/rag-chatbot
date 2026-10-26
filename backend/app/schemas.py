from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import ChatRole


def _normalize_email(value: str) -> str:
    email = value.strip().lower()
    if "@" not in email or email.startswith("@") or " " in email:
        raise ValueError("Invalid email address")
    return email


class SignupRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return _normalize_email(value)


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return _normalize_email(value)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    created_at: datetime


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class ChatSource(BaseModel):
    document_id: str
    source: str
    chunk_index: int
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    session_id: str


class SessionCreate(BaseModel):
    title: str | None = None

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        title = value.strip()
        if len(title) > 120:
            raise ValueError("Title must be at most 120 characters")
        return title


class SessionUpdate(BaseModel):
    title: str

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        title = value.strip()
        if not title:
            raise ValueError("Title cannot be empty")
        if len(title) > 120:
            raise ValueError("Title must be at most 120 characters")
        return title


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: ChatRole
    content: str
    sources: list[ChatSource] | None
    created_at: datetime
