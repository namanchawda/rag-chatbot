"""Shared test setup: hermetic settings, isolated database, and fakes.

No test talks to real Jina or real Neon. The Jina embedding client is either
replaced by a deterministic fake or driven over httpx.MockTransport.
"""

import os
import uuid

import httpx
import pytest

# Must run before any app import so Settings picks these up instead of .env.
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["JWT_SECRET"] = "test-secret-key"
os.environ["JINA_API_KEY"] = "test-jina-key"
os.environ["FRONTEND_ORIGIN"] = "http://localhost:3000"
os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ["GROQ_MODEL"] = "test-groq-model"
# Existing chat endpoint tests stay hermetic: rerank tests opt back in and
# drive the client over httpx.MockTransport.
os.environ["RERANK_ENABLED"] = "false"


from app.db import Base, SessionLocal, engine  # noqa: E402
from app.models import Document, DocumentStatus, User  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings derived from the text length."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0, 0.0]


class FakeVectorStore:
    """Stands in for PGVector; records adds and deletes."""

    def __init__(self) -> None:
        self.added: list[list] = []
        self.deleted: list[list[str]] = []

    def add_documents(self, documents, **kwargs):
        self.added.append(list(documents))
        return [document.id for document in documents]

    def delete(self, ids=None, **kwargs):
        self.deleted.append([str(i) for i in (ids or [])])


@pytest.fixture()
def db_session():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def users(db_session):
    user_a = User(id=uuid.uuid4(), email="a@example.com", password_hash="x")
    user_b = User(id=uuid.uuid4(), email="b@example.com", password_hash="x")
    db_session.add_all([user_a, user_b])
    db_session.commit()
    return {"a": user_a, "b": user_b, "session": db_session}


@pytest.fixture()
def document(db_session, users):
    row = Document(
        id=uuid.uuid4(),
        user_id=users["a"].id,
        filename="notes.txt",
        status=DocumentStatus.PROCESSING,
        chunk_count=0,
        error=None,
    )
    db_session.add(row)
    db_session.commit()
    return row


def make_jina_response(*, data: list, status_code: int = 200) -> httpx.Response:
    import json

    return httpx.Response(
        status_code,
        content=json.dumps({"data": data}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        request=httpx.Request("POST", "https://api.jina.ai/v1/embeddings"),
    )
