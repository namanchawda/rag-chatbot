"""Endpoint tests for /documents with faked auth and a faked vector store."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.deps import get_current_user
from app.main import create_app
from app.models import Document as DocumentRow
from app.models import DocumentStatus
from app.rag import ingest as ingest_module

from .conftest import FakeVectorStore


@pytest.fixture()
def client(users, monkeypatch):
    store = FakeVectorStore()
    monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
    monkeypatch.setattr(
        ingest_module, "delete_document_vectors", lambda s, document_id, n: s.delete(ids=[f"{document_id}:{i}" for i in range(n)])
    )

    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: users["a"]
    with TestClient(app) as test_client:
        yield test_client, store


class TestUpload:
    def test_accepts_upload_and_returns_processing(self, client, users):
        test_client, _ = client
        response = test_client.post(
            "/documents",
            files={"file": ("notes.txt", b"hello content", "text/plain")},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["filename"].endswith("notes.txt") or body["filename"] == "notes.txt"
        assert body["status"] == "processing"
        # BackgroundTasks run before TestClient returns: the row is already done.
        assert body["status"] in {"processing", "ready"}

    def test_rejects_unsupported_extension(self, client, users):
        test_client, _ = client
        response = test_client.post(
            "/documents",
            files={"file": ("report.docx", b"binary", "application/vnd.openxmlformats")},
        )
        assert response.status_code == 415
        assert ".docx" in response.json()["detail"]

    def test_rejects_oversize_upload(self, client, users, monkeypatch):
        test_client, _ = client
        from app.config import get_settings
        from app.routers import documents as documents_router

        tiny_limit = get_settings().model_copy(update={"MAX_UPLOAD_MB": 0})
        monkeypatch.setattr(documents_router, "get_settings", lambda: tiny_limit)
        response = test_client.post(
            "/documents",
            files={"file": ("notes.txt", b"any content", "text/plain")},
        )
        assert response.status_code == 413
        assert "limit" in response.json()["detail"]

    def test_missing_file(self, client, users):
        test_client, _ = client
        response = test_client.post("/documents")
        assert response.status_code == 422


class TestOwnership:
    def test_list_returns_only_caller_documents(self, client, users):
        test_client, _ = client
        mine = DocumentRow(id=uuid.uuid4(), user_id=users["a"].id, filename="mine.txt")
        theirs = DocumentRow(id=uuid.uuid4(), user_id=users["b"].id, filename="theirs.txt")
        users["session"].add_all([mine, theirs])
        users["session"].commit()

        response = test_client.get("/documents")
        names = [row["filename"] for row in response.json()]
        assert names == ["mine.txt"]

    def test_get_other_users_document_is_404(self, client, users):
        test_client, _ = client
        rows = []
        for owner in ("a", "b"):
            row = DocumentRow(id=uuid.uuid4(), user_id=users[owner].id, filename=f"{owner}.txt")
            rows.append(row)
            users["session"].add(row)
        users["session"].commit()

        ok = test_client.get(f"/documents/{rows[0].id}")
        assert ok.status_code == 200
        assert ok.json()["filename"] == "a.txt"

        forbidden = test_client.get(f"/documents/{rows[1].id}")
        assert forbidden.status_code == 404

        bogus = test_client.get("/documents/not-a-uuid")
        assert bogus.status_code == 404

    def test_delete_other_users_document_is_404(self, client, users):
        test_client, _ = client
        blocked = DocumentRow(id=uuid.uuid4(), user_id=users["b"].id, filename="b.txt")
        users["session"].add(blocked)
        users["session"].commit()

        response = test_client.delete(f"/documents/{blocked.id}")
        assert response.status_code == 404
        assert users["session"].get(DocumentRow, blocked.id) is not None


class TestDelete:
    def test_delete_removes_row_and_vectors(self, client, users):
        test_client, store = client
        row = DocumentRow(
            id=uuid.uuid4(),
            user_id=users["a"].id,
            filename="a.txt",
            status=DocumentStatus.READY,
            chunk_count=5,
        )
        users["session"].add(row)
        users["session"].commit()

        response = test_client.delete(f"/documents/{row.id}")
        assert response.status_code == 204
        assert store.deleted == [[f"{row.id}:{i}" for i in range(5)]]
        users["session"].expunge_all()  # avoid the stale cached row
        assert users["session"].get(DocumentRow, row.id) is None
