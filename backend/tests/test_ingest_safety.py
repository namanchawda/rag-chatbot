"""Ingestion safety caps: page limit, empty text, per-document and per-user
chunk limits, batched embedding with rollback. All failures must happen
before the embedder is ever called, and no real Jina/Neon is contacted.
"""

import uuid
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.models import Document as DocumentRow
from app.models import DocumentStatus
from app.rag import ingest as ingest_module
from app.rag.ingest import ingest_document

from .test_rag import _EmbeddingVectorStore, _RecordingEmbeddings, _SessionFactory, _write_temp_pdf


def _patch_caps(monkeypatch, **updates):
    settings = get_settings().model_copy(update=updates)
    monkeypatch.setattr(ingest_module, "get_settings", lambda: settings)
    return settings


class TestPageCap:
    def test_over_cap_fails_before_embedder(self, users, monkeypatch):
        _patch_caps(monkeypatch, MAX_PDF_PAGES=1)
        embeddings = _RecordingEmbeddings()
        store = _EmbeddingVectorStore(embeddings)
        monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
        temp_path = _write_temp_pdf("two pages of prose", pages=2)

        row = DocumentRow(
            id=uuid.uuid4(),
            user_id=users["a"].id,
            filename="big.pdf",
            status=DocumentStatus.PROCESSING,
        )
        users["session"].add(row)
        users["session"].commit()

        ingest_document(
            str(row.id),
            temp_path,
            "big.pdf",
            str(users["a"].id),
            embeddings_factory=lambda: embeddings,
            session_factory=_SessionFactory(users["session"]),
        )

        failed_row = users["session"].get(DocumentRow, row.id)
        assert failed_row.status == DocumentStatus.FAILED
        assert "page" in (failed_row.error or "") and "1" in (failed_row.error or "")
        assert failed_row.chunk_count == 0
        assert embeddings.embed_calls == 0, "no Jina tokens may be spent"
        assert not Path(temp_path).exists()


class TestEmptyText:
    def test_scanned_pdf_fails_before_embedder(self, users, monkeypatch):
        embeddings = _RecordingEmbeddings()
        store = _EmbeddingVectorStore(embeddings)
        monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
        temp_path = _write_temp_pdf(text=None)

        row = DocumentRow(
            id=uuid.uuid4(),
            user_id=users["a"].id,
            filename="scan.pdf",
            status=DocumentStatus.PROCESSING,
        )
        users["session"].add(row)
        users["session"].commit()

        ingest_document(
            str(row.id),
            temp_path,
            "scan.pdf",
            str(users["a"].id),
            embeddings_factory=lambda: embeddings,
            session_factory=_SessionFactory(users["session"]),
        )

        failed_row = users["session"].get(DocumentRow, row.id)
        assert failed_row.status == DocumentStatus.FAILED
        assert failed_row.error == "No readable text found. Scanned PDFs aren't supported."
        assert embeddings.embed_calls == 0
        assert not Path(temp_path).exists()


class TestChunkCap:
    def _seed(self, users, filename="d.pdf"):
        row = DocumentRow(
            id=uuid.uuid4(),
            user_id=users["a"].id,
            filename=filename,
            status=DocumentStatus.PROCESSING,
        )
        users["session"].add(row)
        users["session"].commit()
        return row

    def test_document_chunk_cap_fails_before_embedder(self, users, monkeypatch):
        _patch_caps(monkeypatch, MAX_CHUNKS_PER_DOCUMENT=2)
        embeddings = _RecordingEmbeddings()
        store = _EmbeddingVectorStore(embeddings)
        monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
        monkeypatch.setattr(
            ingest_module,
            "chunk_documents",
            lambda documents, **_kwargs: [Document(page_content=f"chunk {i}") for i in range(3)],
        )
        temp_path = _write_temp_pdf("some text")
        row = self._seed(users)

        ingest_document(
            str(row.id),
            temp_path,
            "d.pdf",
            str(users["a"].id),
            embeddings_factory=lambda: embeddings,
            session_factory=_SessionFactory(users["session"]),
        )

        failed_row = users["session"].get(DocumentRow, row.id)
        assert failed_row.status == DocumentStatus.FAILED
        assert "2" in failed_row.error and "chunk limit per document" in failed_row.error
        assert embeddings.embed_calls == 0
        assert not store.added
        assert not Path(temp_path).exists()

    def test_user_total_cap_fails_before_embedder(self, users, monkeypatch):
        _patch_caps(monkeypatch, MAX_TOTAL_CHUNKS_PER_USER=4)
        embeddings = _RecordingEmbeddings()
        store = _EmbeddingVectorStore(embeddings)
        monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
        monkeypatch.setattr(
            ingest_module,
            "chunk_documents",
            lambda documents, **_kwargs: [Document(page_content=f"chunk {i}") for i in range(3)],
        )
        # An existing READY document already holds 2 of the user's chunks;
        # adding 3 more would break the total of 4.
        users["session"].add(
            DocumentRow(
                id=uuid.uuid4(),
                user_id=users["a"].id,
                filename="already.pdf",
                status=DocumentStatus.READY,
                chunk_count=2,
            )
        )
        temp_path = _write_temp_pdf("some text")
        row = self._seed(users, "new.pdf")

        ingest_document(
            str(row.id),
            temp_path,
            "new.pdf",
            str(users["a"].id),
            embeddings_factory=lambda: embeddings,
            session_factory=_SessionFactory(users["session"]),
        )

        failed_row = users["session"].get(DocumentRow, row.id)
        assert failed_row.status == DocumentStatus.FAILED
        assert "5" in failed_row.error and "chunk limit per user" in failed_row.error
        assert embeddings.embed_calls == 0
        assert not store.added

    def test_other_users_ready_documents_do_not_count(self, users, monkeypatch):
        _patch_caps(monkeypatch, MAX_TOTAL_CHUNKS_PER_USER=4)
        embeddings = _RecordingEmbeddings()
        store = _EmbeddingVectorStore(embeddings)
        monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
        monkeypatch.setattr(
            ingest_module,
            "chunk_documents",
            lambda documents, **_kwargs: [Document(page_content=f"chunk {i}") for i in range(3)],
        )
        # User B's documents are irrelevant to user A's total.
        users["session"].add(
            DocumentRow(
                id=uuid.uuid4(),
                user_id=users["b"].id,
                filename="theirs.pdf",
                status=DocumentStatus.READY,
                chunk_count=9000,
            )
        )
        temp_path = _write_temp_pdf("some text")
        row = self._seed(users, "new.pdf")

        ingest_document(
            str(row.id),
            temp_path,
            "new.pdf",
            str(users["a"].id),
            embeddings_factory=lambda: embeddings,
            session_factory=_SessionFactory(users["session"]),
        )

        row_after = users["session"].get(DocumentRow, row.id)
        assert row_after.status == DocumentStatus.READY
        assert row_after.chunk_count == 3
        assert embeddings.embed_calls > 0


class TestBatchedInsert:
    def _fake_chunks(self, count):
        return [Document(page_content=f"chunk {i}") for i in range(count)]

    def test_insert_runs_in_batches(self, users, monkeypatch):
        monkeypatch.setattr(ingest_module, "_INGEST_BATCH_SIZE", 2)
        embeddings = _RecordingEmbeddings()
        store = _EmbeddingVectorStore(embeddings)
        monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
        monkeypatch.setattr(
            ingest_module, "chunk_documents", lambda documents, **_kwargs: self._fake_chunks(5)
        )
        temp_path = _write_temp_pdf("some text")
        row = DocumentRow(
            id=uuid.uuid4(),
            user_id=users["a"].id,
            filename="d.pdf",
            status=DocumentStatus.PROCESSING,
        )
        users["session"].add(row)
        users["session"].commit()

        ingest_document(
            str(row.id),
            temp_path,
            "d.pdf",
            str(users["a"].id),
            embeddings_factory=lambda: embeddings,
            session_factory=_SessionFactory(users["session"]),
        )

        row_after = users["session"].get(DocumentRow, row.id)
        assert row_after.status == DocumentStatus.READY
        assert [len(batch) for batch in store.added] == [2, 2, 1]
        assert embeddings.embed_calls == 3
        assert store.deleted == [], "nothing to roll back on success"

    def test_batch_failure_deletes_already_inserted_vectors(self, users, monkeypatch):
        monkeypatch.setattr(ingest_module, "_INGEST_BATCH_SIZE", 2)
        embeddings = _RecordingEmbeddings()
        store = _EmbeddingVectorStore(embeddings)

        original_add = store.add_documents

        def add_then_fail(documents, **kwargs):
            if len(store.added) >= 2:
                raise RuntimeError("vector store exploded mid-batch")
            return original_add(documents, **kwargs)

        store.add_documents = add_then_fail
        monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
        monkeypatch.setattr(
            ingest_module, "chunk_documents", lambda documents, **_kwargs: self._fake_chunks(5)
        )
        temp_path = _write_temp_pdf("some text")
        row = DocumentRow(
            id=uuid.uuid4(),
            user_id=users["a"].id,
            filename="d.pdf",
            status=DocumentStatus.PROCESSING,
        )
        users["session"].add(row)
        users["session"].commit()

        ingest_document(
            str(row.id),
            temp_path,
            "d.pdf",
            str(users["a"].id),
            embeddings_factory=lambda: embeddings,
            session_factory=_SessionFactory(users["session"]),
        )

        failed_row = users["session"].get(DocumentRow, row.id)
        assert failed_row.status == DocumentStatus.FAILED
        assert failed_row.chunk_count == 0
        # Two successful batches of 2: exactly those four vector ids deleted.
        assert store.deleted == [[f"{row.id}:{i}" for i in range(4)]]
        assert not Path(temp_path).exists()
