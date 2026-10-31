"""Selectable chunking strategies: unit tests for the splitters plus API,
storage and migration tests. All SQLite / fakes; no real Jina, Groq or Neon.
"""

import os
import tempfile
import uuid

import fitz
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)

from app.config import get_settings
from app.db import migrate_document_columns
from app.deps import get_current_user
from app.main import create_app
from app.models import Document as DocumentRow
from app.rag import ingest as ingest_module
from app.rag.chunking import (
    STRATEGIES,
    UnknownChunkingStrategyError,
    describe_strategies,
    resolve_strategy,
    split_documents,
)

from .conftest import FakeVectorStore

STRATEGY_KEYS = sorted(STRATEGIES.keys())

PYDOC_TEMP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_tmp_should_stay_empty"
)


# --- helpers -----------------------------------------------------------------

def make_doc(text: str, metadata: dict | None = None) -> Document:
    return Document(page_content=text, metadata=metadata or {"source": "doc.txt"})


def sentences(n: int, words: int = 6) -> str:
    """n whole sentences, each ending with a period (indices make them unique)."""
    return " ".join(f"Sentence {i} has {words} words." for i in range(n))


def assert_strategy_invariants(chunks: list[Document], chunk_size: int):
    for chunk in chunks:
        assert len(chunk.page_content) <= chunk_size
        assert chunk.page_content.strip() != ""
        # Source metadata survives every strategy.
        assert "source" in chunk.metadata


# --- strategy unit tests ------------------------------------------------------

class TestStrategyLimits:
    def test_each_strategy_respects_size_limit_and_drops_empty_chunks(self):
        size = get_settings().CHUNK_SIZE_MIN
        overlap = min(50, size // 4)
        # Blank-only gaps on the way: none of that may become a chunk.
        text = "\n\n   \n\n".join(f"Paragraph {i} content. " * 4 for i in range(8))
        doc = make_doc(text, {"source": "report.pdf", "page": 3})
        for key in STRATEGY_KEYS:
            chunks = split_documents([doc], strategy_key=key, chunk_size=size, chunk_overlap=overlap)
            assert chunks, f"strategy {key} produced nothing"
            assert_strategy_invariants(chunks, size)

    def test_sentence_strategy_never_splits_inside_sentences(self):
        body = sentences(12, words=10)
        chunks = split_documents(
            [make_doc(body)], strategy_key="sentence", chunk_size=200, chunk_overlap=40
        )
        assert len(chunks) > 1
        all_chunks_text = "\n---\n".join(chunk.page_content for chunk in chunks)
        # Every whole sentence appears intact inside exactly one chunk: the
        # strategy packs whole sentences and never cuts inside one.
        for index in range(12):
            sentence = f"Sentence {index} has 10 words."
            assert sentence in all_chunks_text
            assert sentence in next(chunk.page_content for chunk in chunks if sentence in chunk.page_content)

    def test_paragraph_strategy_falls_back_for_oversize_paragraph(self):
        big = sentences(20, words=12)
        text = f"{big}\n\nSmall paragraph.\n\nAnother small paragraph."
        chunks = split_documents(
            [make_doc(text)], strategy_key="paragraph", chunk_size=180, chunk_overlap=30
        )
        assert_strategy_invariants(chunks, 180)
        # The oversize paragraph had to be subdivided...
        assert len(chunks) >= 2
        # ...and the small paragraphs survived as their own units.
        assert any("Small paragraph." in chunk.page_content for chunk in chunks)
        assert any("Another small paragraph." in chunk.page_content for chunk in chunks)

    def test_fixed_strategy_uses_windows_ignoring_boundaries(self):
        body = "abcdefghij" * 12  # uniform text, no natural boundaries
        chunks = split_documents(
            [make_doc(body)], strategy_key="fixed", chunk_size=40, chunk_overlap=10
        )
        assert_strategy_invariants(chunks, 40)
        # Windows step by size - overlap = 30, cutting mid-word.
        assert chunks[0].page_content == body[0:40]
        assert chunks[1].page_content == body[30:70]

    def test_fixed_overlap_shares_tail_characters(self):
        body = "abcdefghij" * 12
        chunks = split_documents(
            [make_doc(body)], strategy_key="fixed", chunk_size=40, chunk_overlap=10
        )
        for previous, current in zip(chunks, chunks[1:]):
            assert previous.page_content[-10:] in current.page_content

    def test_sentence_overlap_carries_tail_sentences(self):
        body = sentences(30, words=10)
        chunks = split_documents(
            [make_doc(body)], strategy_key="sentence", chunk_size=150, chunk_overlap=60
        )
        assert len(chunks) > 1
        for previous, current in zip(chunks, chunks[1:]):
            assert len(current.page_content) <= 150
            # The next chunk starts with the trailing sentence(s) that fit the
            # overlap: its first sentence must repeat content of the previous
            # chunk (the shared overlap text), never fresh-only content.
            current_first = current.page_content.split(". ")[0].rstrip(".")
            assert current_first[:20] in previous.page_content

    def test_recursive_is_unchanged_default_behavior(self):
        doc = make_doc(sentences(40, words=8), {"source": "x.pdf", "page": 2})
        chunks = split_documents([doc], strategy_key="recursive", chunk_size=400, chunk_overlap=80)
        expected = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=80, length_function=len
        ).split_documents([make_doc(doc.page_content, doc.metadata)])
        assert [c.page_content for c in chunks] == [c.page_content for c in expected]

    def test_determinism(self):
        body = sentences(25, words=9) + "\n\n" + sentences(6, words=7)
        for key in STRATEGY_KEYS:
            first = split_documents([make_doc(body)], strategy_key=key, chunk_size=220, chunk_overlap=40)
            second = split_documents([make_doc(body)], strategy_key=key, chunk_size=220, chunk_overlap=40)
            assert [c.page_content for c in first] == [c.page_content for c in second]

    def test_metadata_preserved(self):
        metadata = {"source": "p.pdf", "page": 7, "file_path": "/t.pdf"}
        body = sentences(20, words=10)
        for key in STRATEGY_KEYS:
            chunks = split_documents(
                [make_doc(body, metadata)], strategy_key=key, chunk_size=180, chunk_overlap=30
            )
            for chunk in chunks:
                assert chunk.metadata["page"] == 7
                assert chunk.metadata["source"] == "p.pdf"
                assert chunk.metadata["file_path"] == "/t.pdf"

    def test_unknown_strategy_key_is_rejected(self):
        with pytest.raises(UnknownChunkingStrategyError):
            resolve_strategy("semantic")
        with pytest.raises(UnknownChunkingStrategyError):
            split_documents(
                [make_doc("hi")], strategy_key="semantic", chunk_size=300, chunk_overlap=0
            )

    def test_registry_has_labels_and_descriptions(self):
        described = describe_strategies()
        assert [entry["key"] for entry in described] == [
            "recursive",
            "fixed",
            "sentence",
            "paragraph",
        ]
        for entry in described:
            assert entry["label"]
            assert entry["description"]


# --- upload API ---------------------------------------------------------------

@pytest.fixture()
def client(users, monkeypatch):
    store = FakeVectorStore()
    monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
    monkeypatch.setattr(
        ingest_module,
        "delete_document_vectors",
        lambda s, document_id, n: s.delete(ids=[f"{document_id}:{i}" for i in range(n)]),
    )

    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: users["a"]
    with TestClient(app) as test_client:
        yield test_client, store


def pdf_bytes(text: str = "Hello PDF body") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 540, 790), text)
    data = doc.tobytes()
    doc.close()
    return data


class TestUploadValidation:
    @pytest.mark.parametrize(
        "data",
        [
            {"chunking_strategy": "semantic"},
            {"chunk_size": "299"},
            {"chunk_size": "2001"},
            {"chunk_overlap": "-1"},
            {"chunk_overlap": "301"},
            {"chunk_size": "600", "chunk_overlap": "300"},
        ],
    )
    def test_invalid_options_are_422_without_reading_the_file(self, client, monkeypatch, data):
        test_client, _ = client
        monkeypatch.setattr(tempfile, "gettempdir", lambda: PYDOC_TEMP_DIR)
        response = test_client.post(
            "/documents",
            data=data,
            files={"file": ("report.pdf", pdf_bytes(), "application/pdf")},
        )
        assert response.status_code == 422
        # Validation runs before the file is ever read or saved: the temp
        # directory was never created.
        assert not os.path.exists(PYDOC_TEMP_DIR)

    def test_unknown_strategy_422_lists_allowed_keys(self, client):
        test_client, _ = client
        response = test_client.post(
            "/documents",
            data={"chunking_strategy": "semantic"},
            files={"file": ("report.pdf", pdf_bytes(), "application/pdf")},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        for key in STRATEGY_KEYS:
            assert key in detail

    def test_defaults_when_fields_missing(self, client):
        test_client, _ = client
        response = test_client.post(
            "/documents",
            files={"file": ("small.pdf", pdf_bytes(), "application/pdf")},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["chunking_strategy"] == "recursive"
        assert body["chunk_size"] == get_settings().CHUNK_SIZE
        assert body["chunk_overlap"] == get_settings().CHUNK_OVERLAP

    def test_valid_custom_options_are_stored(self, client):
        test_client, _ = client
        response = test_client.post(
            "/documents",
            data={"chunking_strategy": "sentence", "chunk_size": "500", "chunk_overlap": "100"},
            files={"file": ("t.txt", b"Text. " * 200, "text/plain")},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["chunking_strategy"] == "sentence"
        assert body["chunk_size"] == 500
        assert body["chunk_overlap"] == 100


class TestUploadIngestsEveryStrategy:
    @pytest.mark.parametrize("strategy_key", STRATEGY_KEYS)
    def test_pdf_with_each_strategy_ends_ready(self, client, users, strategy_key):
        test_client, store = client
        response = test_client.post(
            "/documents",
            data={"chunking_strategy": strategy_key},
            files={"file": ("book.pdf", pdf_bytes(sentences(30, words=10)), "application/pdf")},
        )
        assert response.status_code == 202
        document_id = response.json()["id"]
        row = users["session"].get(DocumentRow, uuid.UUID(document_id))
        users["session"].refresh(row)
        assert row.status.value == "ready"
        assert row.chunking_strategy == strategy_key
        assert row.chunk_size == get_settings().CHUNK_SIZE
        assert row.chunk_overlap == get_settings().CHUNK_OVERLAP

        all_chunks = [chunk for batch in store.added for chunk in batch]
        assert all_chunks, "vectors must have been added"
        # Chunk ids stay <document_id>:<index>, sequential from 0.
        assert [chunk.id for chunk in all_chunks] == [
            f"{document_id}:{i}" for i in range(len(all_chunks))
        ]
        assert [chunk.metadata["chunk_index"] for chunk in all_chunks] == list(
            range(len(all_chunks))
        )
        assert all(
            chunk.metadata["chunking_strategy"] == strategy_key for chunk in all_chunks
        )

    @pytest.mark.parametrize("strategy_key", STRATEGY_KEYS)
    def test_txt_with_each_strategy_ends_ready(self, client, users, strategy_key):
        test_client, store = client
        content = (sentences(25, words=9) + "\n\n" + sentences(8, words=7)).encode("utf-8")
        response = test_client.post(
            "/documents",
            data={"chunking_strategy": strategy_key},
            files={"file": ("notes.txt", content, "text/plain")},
        )
        assert response.status_code == 202
        document_id = response.json()["id"]
        row = users["session"].get(DocumentRow, uuid.UUID(document_id))
        users["session"].refresh(row)
        assert row.status.value == "ready"
        assert row.chunking_strategy == strategy_key

        all_chunks = [chunk for batch in store.added for chunk in batch]
        assert all_chunks
        assert all(
            chunk.metadata["chunking_strategy"] == strategy_key for chunk in all_chunks
        )
        assert all(
            chunk.metadata["chunk_index"] == index
            for index, chunk in enumerate(all_chunks)
        )

    def test_empty_txt_is_failed_not_ready(self, client, users):
        test_client, _ = client
        response = test_client.post(
            "/documents",
            files={"file": ("empty.txt", b"   \n\n  ", "text/plain")},
        )
        assert response.status_code == 202
        document_id = response.json()["id"]
        row = users["session"].get(DocumentRow, uuid.UUID(document_id))
        users["session"].refresh(row)
        assert row.status.value == "failed"
        assert row.chunk_count == 0


# --- startup migration ---------------------------------------------------------

def _create_old_schema_engine() -> "object":
    """An engine whose documents table predates the chunking columns."""
    engine = create_engine("sqlite://")
    metadata = MetaData()
    Table(
        "documents",
        metadata,
        Column("id", String(36), primary_key=True),
        Column("user_id", String(36)),
        Column("filename", String(255), nullable=False),
        Column("status", String(20), nullable=False),
        Column("chunk_count", Integer, nullable=False),
        Column("error", Text, nullable=True),
        Column("created_at", DateTime),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO documents (id, user_id, filename, status, chunk_count, error, created_at) "
                "VALUES ('11111111-1111-1111-1111-111111111111', "
                "'22222222-2222-2222-2222-222222222222', 'old.pdf', 'ready', 4, NULL, CURRENT_TIMESTAMP)"
            )
        )
    return engine


class TestStartupMigration:
    def test_old_table_gains_columns_and_rows_survive_as_recursive(self):
        engine = _create_old_schema_engine()
        migrate_document_columns(engine)

        with engine.connect() as connection:
            columns = {
                row[1] for row in connection.execute(text("PRAGMA table_info(documents)"))
            }
            assert {"chunking_strategy", "chunk_size", "chunk_overlap"} <= columns
            rows = connection.execute(
                text("SELECT chunking_strategy, chunk_size, chunk_overlap FROM documents")
            ).fetchall()
        assert rows == [("recursive", None, None)]

    def test_migration_is_idempotent_and_runs_twice(self):
        engine = _create_old_schema_engine()
        migrate_document_columns(engine)
        migrate_document_columns(engine)  # second run must be a no-op

        with engine.connect() as connection:
            rows = connection.execute(
                text("SELECT count(*) FROM documents WHERE chunking_strategy = 'recursive'")
            ).scalar()
        assert rows == 1

    def test_new_table_needs_no_migration(self, db_session):
        # The current ORM table already carries the columns: a no-op that
        # must not raise.
        migrate_document_columns(db_session.bind)
