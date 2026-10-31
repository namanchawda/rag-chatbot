"""Chunks browser API tests.

The production tables are langchain_postgres' own (see
app/rag/chunks.py docstring for the real column names read from
langchain_postgres 0.0.19 - id, collection_id, embedding, document,
cmetadata / uuid, name, cmetadata). SQLite cannot hold a pgvector Vector
column, and langchain_postgres' JSONB model is Postgres-only, so the tests
build a minimal mirror schema with those exact column names (generic JSON
instead of JSONB, no embedding column) and seed it directly. No real Neon,
Jina or Groq is contacted.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert

from app.db import SessionLocal, engine
from app.deps import get_current_user
from app.main import create_app
from app.models import Document, DocumentStatus
from app.rag import chunks as chunks_store
from app.rag.ingest import COLLECTION_NAME

COLLECTION_UUID = str(uuid.UUID(int=1234))


@pytest.fixture()
def chunk_mirror():
    """The mirrored langchain tables on SQLite, only for these tests."""
    chunks_store.chunks_metadata.create_all(bind=engine)
    with SessionLocal() as db:
        db.execute(
            insert(chunks_store.collection_table).values(
                uuid=COLLECTION_UUID, name=COLLECTION_NAME, cmetadata={}
            )
        )
        db.commit()
    yield
    chunks_store.chunks_metadata.drop_all(bind=engine)


def _seed_chunk(
    owner_id: str,
    doc_id: str,
    source: str,
    index: int,
    text: str,
) -> str:
    """Insert one embedding row exactly as ingest._annotate would store it."""
    chunk_id = f"{doc_id}:{index}"
    with SessionLocal() as db:
        db.execute(
            insert(chunks_store.embedding_table).values(
                id=chunk_id,
                collection_id=COLLECTION_UUID,
                document=text,
                cmetadata={
                    "user_id": owner_id,
                    "document_id": doc_id,
                    "source": source,
                    "chunk_index": index,
                },
            )
        )
        db.commit()
    return chunk_id


def _seed_document(db_session, owner_id, doc_id, filename="d.pdf", status=DocumentStatus.READY):
    row = Document(id=uuid.UUID(doc_id), user_id=owner_id, filename=filename, status=status)
    db_session.add(row)
    db_session.commit()
    return row


def _client(monkeypatch, user) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: user
    return TestClient(app)


def _doc_id(seed: int) -> str:
    return str(uuid.UUID(int=seed))


class TestPagination:
    def test_default_limit_twenty_and_useful_counts(
        self, db_session, users, chunk_mirror, monkeypatch
    ):
        doc = _doc_id(101)
        _seed_document(db_session, users["a"].id, doc)
        for index in range(25):
            _seed_chunk(str(users["a"].id), doc, "d.pdf", index, f"text {index}")

        with _client(monkeypatch, users["a"]) as client:
            default_page = client.get("/chunks").json()
        assert default_page["limit"] == 20
        assert default_page["offset"] == 0
        assert default_page["total"] == 25
        assert len(default_page["items"]) == 20

        with _client(monkeypatch, users["a"]) as client:
            tail = client.get("/chunks", params={"offset": 20}).json()
        assert tail["total"] == 25
        assert len(tail["items"]) == 5

    def test_limit_cap_fifty(self, db_session, users, chunk_mirror, monkeypatch):
        doc = _doc_id(102)
        _seed_document(db_session, users["a"].id, doc)
        for index in range(25):
            _seed_chunk(str(users["a"].id), doc, "d.pdf", index, f"text {index}")

        with _client(monkeypatch, users["a"]) as client:
            full = client.get("/chunks", params={"limit": 50}).json()
            assert len(full["items"]) == 25
            assert full["limit"] == 50
            over = client.get("/chunks", params={"limit": 51})
            assert over.status_code == 422
            zero = client.get("/chunks", params={"limit": 0})
            assert zero.status_code == 422
            negative_offset = client.get("/chunks", params={"offset": -1})
            assert negative_offset.status_code == 422

    def test_ordering_filename_then_numeric_chunk_index(
        self, db_session, users, chunk_mirror, monkeypatch
    ):
        doc_b = _doc_id(201)
        doc_a = _doc_id(202)
        _seed_document(db_session, users["a"].id, doc_a, filename="a.pdf")
        _seed_document(db_session, users["a"].id, doc_b, filename="b.pdf")
        owner = str(users["a"].id)
        for index in (10, 0, 2):
            _seed_chunk(owner, doc_b, "b.pdf", index, f"b text {index}")
            _seed_chunk(owner, doc_a, "a.pdf", index, f"a text {index}")

        with _client(monkeypatch, users["a"]) as client:
            items = client.get("/chunks", params={"limit": 50}).json()["items"]

        # Numeric order: chunk 10 comes after 2, never between 0 and 2.
        assert [
            (item["source"], item["chunk_index"]) for item in items
        ] == [
            ("a.pdf", 0),
            ("a.pdf", 2),
            ("a.pdf", 10),
            ("b.pdf", 0),
            ("b.pdf", 2),
            ("b.pdf", 10),
        ]


class TestFields:
    def test_item_shape_and_never_the_embedding(
        self, db_session, users, chunk_mirror, monkeypatch
    ):
        doc = _doc_id(301)
        _seed_document(db_session, users["a"].id, doc)
        chunk_id = _seed_chunk(str(users["a"].id), doc, "d.pdf", 3, "x" * 100)

        with _client(monkeypatch, users["a"]) as client:
            body = client.get("/chunks", params={"limit": 1}).json()
            single = client.get(f"/chunks/{chunk_id}").json()

        assert set(body) == {"items", "total", "limit", "offset"}
        assert list(body["items"][0]) == [
            "id",
            "document_id",
            "source",
            "chunk_index",
            "text",
            "char_count",
            "token_estimate",
        ]
        item = body["items"][0]
        assert item["id"] == chunk_id
        assert item["document_id"] == doc
        assert item["source"] == "d.pdf"
        assert item["char_count"] == 100
        assert item["token_estimate"] == 100 // 4
        assert single["text"] == "x" * 100
        assert "embedding" not in single

    def test_single_chunk_missing_is_404(self, db_session, users, chunk_mirror, monkeypatch):
        with _client(monkeypatch, users["a"]) as client:
            missing = client.get("/chunks/nonexistent-id")
        assert missing.status_code == 404
        assert missing.json()["detail"] == "Chunk not found"


class TestDocumentFilter:
    def test_document_id_scopes_to_owned_document(
        self, db_session, users, chunk_mirror, monkeypatch
    ):
        mine = _doc_id(401)
        other_of_mine = _doc_id(402)
        _seed_document(db_session, users["a"].id, mine, filename="mine.pdf")
        _seed_document(db_session, users["a"].id, other_of_mine, filename="also-mine.pdf")
        owner = str(users["a"].id)
        _seed_chunk(owner, mine, "mine.pdf", 0, "first")
        _seed_chunk(owner, other_of_mine, "also-mine.pdf", 0, "second")

        with _client(monkeypatch, users["a"]) as client:
            body = client.get("/chunks", params={"document_id": mine}).json()

        assert body["total"] == 1
        assert body["items"][0]["document_id"] == mine
        assert body["items"][0]["text"] == "first"

    def test_foreign_or_unknown_document_id_is_404(
        self, db_session, users, chunk_mirror, monkeypatch
    ):
        b_doc = _doc_id(403)
        _seed_document(db_session, users["b"].id, b_doc)
        _seed_chunk(str(users["b"].id), b_doc, "b.pdf", 0, "b content")

        with _client(monkeypatch, users["a"]) as client:
            foreign = client.get("/chunks", params={"document_id": b_doc})
            unknown = client.get("/chunks", params={"document_id": _doc_id(404)})
            malformed = client.get("/chunks", params={"document_id": "not-a-uuid"})

        assert foreign.status_code == 404
        assert unknown.status_code == 404
        assert malformed.status_code == 404


class TestSearch:
    def test_special_characters_match_literally(
        self, db_session, users, chunk_mirror, monkeypatch
    ):
        doc = _doc_id(501)
        _seed_document(db_session, users["a"].id, doc)
        owner = str(users["a"].id)
        _seed_chunk(owner, doc, "d.pdf", 0, "a_b and 100% sold")
        _seed_chunk(owner, doc, "d.pdf", 1, "ab plain, nothing special")
        _seed_chunk(owner, doc, "d.pdf", 2, "he said 'quote me' twice")

        with _client(monkeypatch, users["a"]) as client:
            underscore = client.get("/chunks", params={"q": "a_b"}).json()
            # An unescaped "_" is a LIKE wildcard for any character.
            underscore_alone = client.get("/chunks", params={"q": "_"}).json()
            percent = client.get("/chunks", params={"q": "0%"}).json()
            quote = client.get("/chunks", params={"q": "'quote me'"}).json()
            no_match = client.get("/chunks", params={"q": "a b"}).json()

        assert [i["chunk_index"] for i in underscore["items"]] == [0]
        assert underscore_alone["total"] == 1
        assert [i["chunk_index"] for i in percent["items"]] == [0]
        assert [i["chunk_index"] for i in quote["items"]] == [2]
        assert no_match["total"] == 0

    def test_q_is_case_insensitive(self, db_session, users, chunk_mirror, monkeypatch):
        doc = _doc_id(502)
        _seed_document(db_session, users["a"].id, doc)
        _seed_chunk(str(users["a"].id), doc, "d.pdf", 0, "Mixed CASE text")

        with _client(monkeypatch, users["a"]) as client:
            hit = client.get("/chunks", params={"q": "mixed case"}).json()
            capped = client.get("/chunks", params={"q": "x" * 201})
        assert hit["total"] == 1
        assert capped.status_code == 422


class TestUserIsolation:
    def _mixed_documents(self, db_session):
        a_doc = _doc_id(601)
        b_doc = _doc_id(602)
        return a_doc, b_doc

    def test_list_total_single_and_stats_never_cross_users(
        self, db_session, users, chunk_mirror, monkeypatch
    ):
        a_doc, b_doc = self._mixed_documents(db_session)
        _seed_document(db_session, users["a"].id, a_doc)
        _seed_document(db_session, users["b"].id, b_doc)
        a_chunk = _seed_chunk(str(users["a"].id), a_doc, "a.pdf", 0, "secret of a")
        _seed_chunk(str(users["b"].id), b_doc, "b.pdf", 0, "secret of b")

        with _client(monkeypatch, users["a"]) as client:
            listing = client.get("/chunks").json()
            stats = client.get("/chunks/stats").json()
            foreign_chunk = client.get(f"/chunks/{a_chunk}")  # same client sees own
        assert listing["total"] == 1
        assert listing["items"][0]["text"] == "secret of a"
        assert stats == {"documents": 1, "chunks": 1, "characters": len("secret of a")}
        assert foreign_chunk.status_code == 200

        with _client(monkeypatch, users["b"]) as client:
            b_listing = client.get("/chunks").json()
            b_stats = client.get("/chunks/stats").json()
            foreign_single = client.get(f"/chunks/{a_chunk}")
        assert b_listing["total"] == 1
        assert [i["text"] for i in b_listing["items"]] == ["secret of b"]
        assert b_stats == {"documents": 1, "chunks": 1, "characters": len("secret of b")}
        assert foreign_single.status_code == 404, b"foreign chunk must be 404, not 403"
        assert foreign_single.json()["detail"] == "Chunk not found"

    def test_stats_count_only_ready_documents(
        self, db_session, users, chunk_mirror, monkeypatch
    ):
        ready_doc = _doc_id(603)
        processing_doc = _doc_id(604)
        _seed_document(db_session, users["a"].id, ready_doc, filename="ready.pdf")
        _seed_document(
            db_session, users["a"].id, processing_doc, status=DocumentStatus.PROCESSING
        )
        owner = str(users["a"].id)
        _seed_chunk(owner, ready_doc, "ready.pdf", 0, "four")
        _seed_chunk(owner, processing_doc, "processing.pdf", 0, "not counted at all")

        with _client(monkeypatch, users["a"]) as client:
            stats = client.get("/chunks/stats").json()

        assert stats == {
            "documents": 1,
            "chunks": 1,
            "characters": len("four"),
        }


class TestEmptyDatabase:
    """No langchain tables (brand-new database, no ingestion): empty, not 500."""

    def test_list_stats_and_single_without_tables(
        self, db_session, users, monkeypatch
    ):
        with _client(monkeypatch, users["a"]) as client:
            listing = client.get("/chunks").json()
            stats = client.get("/chunks/stats").json()
            missing = client.get("/chunks/some-id")
        assert listing == {"items": [], "total": 0, "limit": 20, "offset": 0}
        assert stats == {"documents": 0, "chunks": 0, "characters": 0}
        assert missing.status_code == 404

    def test_list_after_zero_chunks_ingested(
        self, db_session, users, chunk_mirror, monkeypatch
    ):
        with _client(monkeypatch, users["a"]) as client:
            listing = client.get("/chunks").json()
        assert listing["total"] == 0
        assert listing["items"] == []
