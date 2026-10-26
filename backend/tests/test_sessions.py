"""Session CRUD tests: ownership, ordering, caps, and cascade delete.

No LLM or vector store involved; sessions are seeded directly.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import SessionLocal
from app.deps import get_current_user
from app.main import create_app
from app.models import ChatMessage, ChatRole, ChatSession


def _client(monkeypatch, user):
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: user
    return TestClient(app)


@pytest.fixture()
def session_row(db_session, users):
    row = ChatSession(id=uuid.uuid4(), user_id=users["a"].id, title="existing")
    db_session.add(row)
    db_session.commit()
    return row


class TestCreateAndList:
    def test_create_returns_session_with_blank_title(self, monkeypatch, users):
        with _client(monkeypatch, users["a"]) as client:
            response = client.post("/sessions", json={})
        assert response.status_code == 201
        body = response.json()
        assert body["title"] == ""
        assert body["id"]

    def test_create_with_title_is_trimmed(self, monkeypatch, users):
        with _client(monkeypatch, users["a"]) as client:
            response = client.post("/sessions", json={"title": "  hello world  "})
        assert response.status_code == 201
        assert response.json()["title"] == "hello world"

    def test_create_title_too_long(self, monkeypatch, users):
        with _client(monkeypatch, users["a"]) as client:
            response = client.post("/sessions", json={"title": "x" * 121})
        assert response.status_code == 422

    def test_list_most_recent_first_with_limit(self, monkeypatch, users, db_session):
        now = datetime.now(timezone.utc)
        rows = [
            ChatSession(
                id=uuid.uuid4(),
                user_id=users["a"].id,
                title=f"s{i}",
                created_at=now - timedelta(minutes=4 - i),
                updated_at=now - timedelta(minutes=4 - i),
            )
            for i in range(4)
        ]
        db_session.add_all(rows)
        db_session.commit()
        with _client(monkeypatch, users["a"]) as client:
            listing = client.get("/sessions", params={"limit": 2}).json()
            full = client.get("/sessions").json()
        assert [item["title"] for item in listing] == ["s3", "s2"]
        assert len(full) == 4

    def test_list_cap_at_100(self, monkeypatch, users, db_session):
        db_session.add_all(
            [
                ChatSession(id=uuid.uuid4(), user_id=users["a"].id, title=f"s{i}")
                for i in range(105)
            ]
        )
        db_session.commit()
        with _client(monkeypatch, users["a"]) as client:
            listing = client.get("/sessions", params={"limit": 150}).json()
        assert len(listing) == 100

    def test_list_excludes_other_users(self, monkeypatch, users, db_session):
        db_session.add_all(
            [
                ChatSession(id=uuid.uuid4(), user_id=users["b"].id, title="b"),
                ChatSession(id=uuid.uuid4(), user_id=users["a"].id, title="a"),
            ]
        )
        db_session.commit()
        with _client(monkeypatch, users["a"]) as client:
            listing = client.get("/sessions").json()
        assert [item["title"] for item in listing] == ["a"]


class TestListMessages:
    def test_messages_in_order_oldest_first(self, monkeypatch, users, session_row, db_session):
        base = datetime.now(timezone.utc)
        roles = [ChatRole.USER, ChatRole.ASSISTANT, ChatRole.USER]
        messages = [
            ChatMessage(
                id=uuid.uuid4(),
                session_id=session_row.id,
                role=roles[index],
                content=f"m{index}",
                created_at=base + timedelta(seconds=index),
            )
            for index in range(3)
        ]
        db_session.add_all(messages)
        db_session.commit()
        with _client(monkeypatch, users["a"]) as client:
            rows = client.get(f"/sessions/{session_row.id}/messages").json()
        assert [row["content"] for row in rows] == ["m0", "m1", "m2"]
        assert rows[0]["role"] == "user"
        assert rows[1]["role"] == "assistant"
        assert rows[0]["sources"] is None

    def test_messages_cap_at_500(self, monkeypatch, users, session_row, db_session):
        base = datetime.now(timezone.utc)
        messages = [
            ChatMessage(
                id=uuid.uuid4(),
                session_id=session_row.id,
                role=ChatRole.USER,
                content=f"m{index}",
                created_at=base + timedelta(milliseconds=index),
            )
            for index in range(505)
        ]
        db_session.add_all(messages)
        db_session.commit()
        with _client(monkeypatch, users["a"]) as client:
            rows = client.get(f"/sessions/{session_row.id}/messages").json()
        assert len(rows) == 500
        contents = [row["content"] for row in rows]
        assert contents[0] == "m5"
        assert contents[-1] == "m504"

    def test_invalid_session_id_is_404(self, monkeypatch, users):
        with _client(monkeypatch, users["a"]) as client:
            response = client.get("/sessions/not-a-uuid/messages")
        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"


class TestRename:
    def test_patch_updates_and_trims_title(self, monkeypatch, users, session_row):
        with _client(monkeypatch, users["a"]) as client:
            response = client.patch(
                f"/sessions/{session_row.id}", json={"title": "  renamed  "}
            )
        assert response.status_code == 200
        assert response.json()["title"] == "renamed"

    def test_patch_empty_title_422(self, monkeypatch, users, session_row):
        with _client(monkeypatch, users["a"]) as client:
            response = client.patch(f"/sessions/{session_row.id}", json={"title": "   "})
        assert response.status_code == 422

    def test_patch_too_long_title_422(self, monkeypatch, users, session_row):
        with _client(monkeypatch, users["a"]) as client:
            response = client.patch(
                f"/sessions/{session_row.id}", json={"title": "x" * 121}
            )
        assert response.status_code == 422


class TestDeleteCascade:
    def test_delete_removes_session_and_messages(self, monkeypatch, users, session_row, db_session):
        db_session.add_all(
            [
                ChatMessage(
                    id=uuid.uuid4(),
                    session_id=session_row.id,
                    role=ChatRole.USER,
                    content="hi",
                ),
                ChatMessage(
                    id=uuid.uuid4(),
                    session_id=session_row.id,
                    role=ChatRole.ASSISTANT,
                    content="hello",
                ),
            ]
        )
        db_session.commit()
        with _client(monkeypatch, users["a"]) as client:
            response = client.delete(f"/sessions/{session_row.id}")
        assert response.status_code == 204
        with SessionLocal() as db:
            assert db.get(ChatSession, session_row.id) is None
            assert (
                db.scalars(
                    select(ChatMessage).where(ChatMessage.session_id == session_row.id)
                ).all()
                == []
            )

    def test_delete_is_only_once(self, monkeypatch, users, session_row):
        with _client(monkeypatch, users["a"]) as client:
            first = client.delete(f"/sessions/{session_row.id}")
            second = client.delete(f"/sessions/{session_row.id}")
        assert first.status_code == 204
        assert second.status_code == 404


class TestCrossUserAccess:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/sessions/{sid}/messages"),
            ("patch", "/sessions/{sid}"),
            ("delete", "/sessions/{sid}"),
        ],
    )
    def test_other_user_gets_404(self, monkeypatch, users, session_row, method, path):
        with _client(monkeypatch, users["b"]) as client:
            kwargs = {"json": {"title": "x"}} if method == "patch" else {}
            response = getattr(client, method)(path.format(sid=session_row.id), **kwargs)
        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"
