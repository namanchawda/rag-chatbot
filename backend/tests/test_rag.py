"""Tests for loaders, splitting, metadata, and the ingest job, plus
JinaEmbeddings behaviour over httpx.MockTransport (no network)."""

import json
import os
import tempfile
import uuid
from pathlib import Path

import httpx
import pytest
from langchain_core.documents import Document

from app.config import get_settings
from app.models import Document as DocumentRow
from app.models import DocumentStatus
from app.rag import ingest as ingest_module
from app.rag.embeddings import JinaEmbeddingError, JinaEmbeddings
from app.rag.ingest import (
    UnsupportedFileTypeError,
    chunk_documents,
    ingest_document,
    load_document,
)

from .conftest import FakeEmbeddings, FakeVectorStore, make_jina_response


def _write_temp_pdf(text: str | None = "Hello PDF body", pages: int = 1) -> str:
    """A real PDF written to a temp path, optionally textless or multi-page."""
    import fitz

    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        if text:
            # insert_text (not insert_textbox): long single-line text never
            # overflows the box and silently disappears.
            page.insert_text((50, 100), text, fontsize=6)
    descriptor, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(descriptor)
    doc.save(temp_path)
    doc.close()
    return temp_path


class _RecordingEmbeddings(FakeEmbeddings):
    """Counts embed calls so tests can prove no tokens were spent."""

    def __init__(self) -> None:
        self.embed_calls = 0

    def embed_documents(self, texts):
        self.embed_calls += 1
        return super().embed_documents(texts)


class _EmbeddingVectorStore(FakeVectorStore):
    """Fake PGVector that, like the real one, embeds before storing."""

    def __init__(self, embeddings) -> None:
        super().__init__()
        self.embeddings = embeddings

    def add_documents(self, documents, **kwargs):
        if self.embeddings is not None:
            self.embeddings.embed_documents([document.page_content for document in documents])
        return super().add_documents(documents, **kwargs)


class TestLoaders:
    def test_load_pdf(self):
        temp_path = _write_temp_pdf("Real PDF text from fitz")
        try:
            documents = load_document(temp_path, "report.pdf")
            text = " ".join(document.page_content for document in documents)
            assert "Real PDF text" in text
        finally:
            Path(temp_path).unlink()

    def test_load_rejects_unsupported_type(self):
        path = _write_temp_pdf()
        try:
            with pytest.raises(UnsupportedFileTypeError) as error:
                load_document(path, "file.docx")
            assert ".docx" in str(error.value)
            assert ".pdf" in str(error.value)
        finally:
            Path(path).unlink()

    def test_load_rejects_md_and_html(self):
        path = _write_temp_pdf()
        try:
            for filename in ("readme.md", "page.html"):
                with pytest.raises(UnsupportedFileTypeError) as error:
                    load_document(path, filename)
                assert filename.rsplit(".", 1)[-1] in str(error.value)
        finally:
            Path(path).unlink()

    def test_load_accepts_txt_with_utf8_content(self):
        descriptor, path = tempfile.mkstemp(suffix=".txt")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write("Bonjour, this is plain text.".encode("utf-8"))
            documents = load_document(path, "notes.txt")
            assert len(documents) == 1
            assert documents[0].page_content == "Bonjour, this is plain text."
            assert "source" in documents[0].metadata
        finally:
            Path(path).unlink()

    def test_load_rejects_non_utf8_txt(self):
        descriptor, path = tempfile.mkstemp(suffix=".txt")
        os.close(descriptor)
        try:
            with open(path, "wb") as handle:
                handle.write(b"\xff\xfe\x00\x01invalid")
            with pytest.raises(ingest_module.NoTextContentError) as error:
                load_document(path, "notes.txt")
            assert "UTF-8" in str(error.value)
        finally:
            Path(path).unlink()

    def test_page_cap_fails_with_friendly_error(self, monkeypatch):
        settings = get_settings().model_copy(update={"MAX_PDF_PAGES": 1})
        monkeypatch.setattr(ingest_module, "get_settings", lambda: settings)
        temp_path = _write_temp_pdf("page one and two", pages=2)
        try:
            with pytest.raises(ingest_module.PdfTooManyPagesError) as error:
                load_document(temp_path, "big.pdf")
            assert "2 pages" in str(error.value)
            assert "1 page limit" in str(error.value)
            assert "Traceback" not in str(error.value)
        finally:
            Path(temp_path).unlink()

    def test_scanned_pdf_fails_with_friendly_error(self):
        temp_path = _write_temp_pdf(text=None)
        try:
            with pytest.raises(ingest_module.ScannedPdfError) as error:
                load_document(temp_path, "scan.pdf")
            assert str(error.value) == "No readable text found. Scanned PDFs aren't supported."
        finally:
            Path(temp_path).unlink()

    def test_encrypted_pdf_fails_with_friendly_error(self):
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 540, 790), "secret")
        descriptor, temp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(descriptor)
        doc.save(
            temp_path,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="o",
            user_pw="u",
            permissions=-1,
        )
        doc.close()
        try:
            probe = fitz.open(temp_path)
            assert probe.needs_pass
            probe.close()
            with pytest.raises(ingest_module.EncryptedPdfError) as error:
                load_document(temp_path, "locked.pdf")
            assert "password" in str(error.value)
            assert "Traceback" not in str(error.value)
        finally:
            Path(temp_path).unlink()


class TestSplitterAndMetadata:
    def test_chunking_respects_size_and_overlap(self):
        long_text = "The quick brown fox jumps over the lazy dog. " * 120
        chunks = chunk_documents([Document(page_content=long_text)])

        assert len(chunks) > 1
        settings = get_settings()
        for chunk in chunks:
            assert len(chunk.page_content) <= settings.CHUNK_SIZE
        # Chunks must overlap: the tail of one chunk appears in the next.
        tail = chunks[0].page_content[-40:]
        assert tail in chunks[1].page_content

    def test_annotate_metadata_and_deterministic_id(self):
        chunk = Document(page_content="chunk text")
        annotated = ingest_module._annotate(
            chunk,
            user_id="user-1",
            document_id="doc-9",
            source="notes.txt",
            index=3,
            strategy="recursive",
        )
        assert chunk.metadata["user_id"] == "user-1"
        assert chunk.metadata["document_id"] == "doc-9"
        assert chunk.metadata["source"] == "notes.txt"
        assert chunk.metadata["chunk_index"] == 3
        assert chunk.id == "doc-9:3"


class _SessionFactory:
    """Returns the same session the test prepared."""

    def __init__(self, session):
        self._session = session

    def __call__(self):
        return self._session


class TestIngestJob:
    def test_success_marks_ready_and_deletes_temp(self, users, monkeypatch):
        embeddings = _RecordingEmbeddings()
        store = _EmbeddingVectorStore(embeddings)
        monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
        temp_path = _write_temp_pdf("alpha beta " * 500)
        assert Path(temp_path).exists()

        document_id = uuid.uuid4()
        users["session"].add(
            DocumentRow(
                id=document_id,
                user_id=users["a"].id,
                filename="notes.pdf",
                status=DocumentStatus.PROCESSING,
            )
        )
        users["session"].commit()
        ingest_document(
            str(document_id),
            temp_path,
            "notes.pdf",
            str(users["a"].id),
            embeddings_factory=lambda: embeddings,
            session_factory=_SessionFactory(users["session"]),
        )

        row = users["session"].get(DocumentRow, document_id)
        assert row is not None
        assert row.status == DocumentStatus.READY
        assert row.chunk_count > 0
        assert row.error is None
        assert store.added, "chunks were not added to the store"
        assert embeddings.embed_calls == row.chunk_count > 0
        first_chunk = store.added[0][0]
        assert first_chunk.metadata["document_id"] == str(document_id)
        assert first_chunk.metadata["source"] == "notes.pdf"
        assert first_chunk.metadata["chunk_index"] == 0
        assert not Path(temp_path).exists(), "temp file was not deleted after ingestion"

    def test_failure_marks_failed_and_still_deletes_temp(self, users, monkeypatch):
        embeddings = _RecordingEmbeddings()
        store = _EmbeddingVectorStore(embeddings)
        monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
        temp_path = _write_temp_pdf("will never load")

        row = DocumentRow(
            id=uuid.uuid4(),
            user_id=users["a"].id,
            filename="broken.docx",
            status=DocumentStatus.PROCESSING,
        )
        users["session"].add(row)
        users["session"].commit()

        ingest_document(
            str(row.id),
            temp_path,
            "broken.docx",
            str(users["a"].id),
            embeddings_factory=lambda: embeddings,
            session_factory=_SessionFactory(users["session"]),
        )

        failed_row = users["session"].get(DocumentRow, row.id)
        assert failed_row.status == DocumentStatus.FAILED
        assert failed_row.chunk_count == 0
        assert "Unsupported file type" in (failed_row.error or "")
        assert not store.added, "nothing should reach the store after loading fails"
        assert embeddings.embed_calls == 0, "the embedder must never run after loading fails"
        assert not Path(temp_path).exists(), "temp file must be deleted even when ingestion fails"


class TestJinaEmbeddings:
    def _client(self, handler) -> JinaEmbeddings:
        instance = JinaEmbeddings(
            api_key="test-key", model="jina-embeddings-v5-omni-small", dimensions=3
        )
        instance._client = httpx.Client(
            transport=httpx.MockTransport(handler),
            timeout=httpx.Timeout(60.0),
        )
        return instance

    def test_missing_key_raises_clear_error(self):
        with pytest.raises(JinaEmbeddingError, match="JINA_API_KEY"):
            JinaEmbeddings(api_key="", model="m", dimensions=3)

    def test_query_body_shape_and_header(self):
        request_box: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            request_box["body"] = json.loads(request.read().decode("utf-8"))
            request_box["auth"] = request.headers["Authorization"]
            return make_jina_response(data=[{"index": 0, "embedding": [0.1, 0.2, 0.3]}])

        vector = self._client(handler).embed_query("question text")

        assert vector == [0.1, 0.2, 0.3]
        body = request_box["body"]
        assert body["model"] == "jina-embeddings-v5-omni-small"
        assert body["task"] == "retrieval.query"
        assert body["normalized"] is True
        assert body["dimensions"] == 3
        assert body["input"] == [{"text": "question text"}]
        assert request_box["auth"].startswith("Bearer test-jina-key") or request_box[
            "auth"
        ] == "Bearer test-key"

    def test_documents_use_passage_task_and_batch_by_32(self):
        batches: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.read().decode("utf-8"))
            count = len(payload["input"])
            batches.append(payload)
            data = [
                {"index": i, "embedding": [float(i), 0.0, 1.0]} for i in range(count)
            ]
            return make_jina_response(data=list(reversed(data)))

        texts = [f"text {i}" for i in range(70)]
        vectors = self._client(handler).embed_documents(texts)

        assert [len(batch["input"]) for batch in batches] == [32, 32, 6]
        assert all(batch["task"] == "retrieval.passage" for batch in batches)
        assert len(vectors) == 70
        assert vectors[0] == [0.0, 0.0, 1.0]
        assert vectors[34] == [2.0, 0.0, 1.0]
        assert vectors[69] == [5.0, 0.0, 1.0]

    def test_input_items_are_objects_with_text_key(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["input"] = json.loads(request.read().decode("utf-8"))["input"]
            return make_jina_response(
                data=[{"index": i, "embedding": [0.0, 0.0, 0.0]} for i in range(2)]
            )

        self._client(handler).embed_documents(["a", "b"])
        assert all(isinstance(item, dict) and "text" in item for item in seen["input"])

    def test_dimension_mismatch_raises_clear_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return make_jina_response(data=[{"index": 0, "embedding": [1.0, 0.0]}])

        with pytest.raises(JinaEmbeddingError, match="EMBEDDING_DIMENSION"):
            self._client(handler).embed_query("hi")

    def test_retries_429_then_succeeds(self):
        state = {"calls": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["calls"] += 1
            if state["calls"] == 1:
                return httpx.Response(
                    429,
                    content=b'{"error": "rate limited"}',
                    headers={"Retry-After": "0"},
                    request=httpx.Request("POST", "https://api.jina.ai/v1/embeddings"),
                )
            return make_jina_response(
                data=[{"index": 0, "embedding": [1.0, 0.0, 1.0]}]
            )

        vector = self._client(handler).embed_query("retry me")
        assert vector == [1.0, 0.0, 1.0]
        assert state["calls"] == 2

    def test_rejected_task_value_surfaces_verbatim_without_retry(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                content=(
                    b'{"error": {"message": "Unsupported task value: retrieval.query.exotic"}}'
                ),
                request=httpx.Request("POST", "https://api.jina.ai/v1/embeddings"),
            )

        with pytest.raises(JinaEmbeddingError, match="retrieval.query.exotic"):
            self._client(handler).embed_query("hello")
