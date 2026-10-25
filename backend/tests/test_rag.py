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


def _write_temp(suffix: str, content: bytes | str) -> str:
    descriptor, temp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(descriptor, "wb") as handle:
        if isinstance(content, bytes):
            handle.write(content)
        else:
            handle.write(content.encode("utf-8"))
    return temp_path


class TestLoaders:
    def test_load_txt(self):
        path = _write_temp(".txt", "hello world")
        try:
            documents = load_document(path, "notes.txt")
            assert len(documents) == 1
            assert "hello world" in documents[0].page_content
        finally:
            Path(path).unlink()

    def test_load_md_falls_back_on_encoding(self):
        latin = b"caf\xe9 r\xe9sum\xe9 " * 20  # valid latin-1, invalid UTF-8
        path = _write_temp(".md", latin)
        try:
            documents = load_document(path, "readme.md")
            assert documents
            assert "caf" in documents[0].page_content
        finally:
            Path(path).unlink()

    def test_load_html(self):
        path = _write_temp(
            ".html", "<html><body><p>Document<i>body text</i></p></body></html>"
        )
        try:
            documents = load_document(path, "page.html")
            text = " ".join(document.page_content for document in documents)
            assert "Document" in text
        finally:
            Path(path).unlink()

    def test_load_pdf(self):
        import fitz

        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Real PDF text from fitz")
        descriptor, temp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(descriptor)
        pdf.save(temp_path)
        pdf.close()
        try:
            documents = load_document(temp_path, "report.pdf")
            text = " ".join(document.page_content for document in documents)
            assert "Real PDF text" in text
        finally:
            Path(temp_path).unlink()

    def test_load_rejects_unsupported_type(self):
        path = _write_temp(".docx", "not really docx")
        try:
            with pytest.raises(UnsupportedFileTypeError) as error:
                load_document(path, "file.docx")
            assert ".docx" in str(error.value)
        finally:
            Path(path).unlink()


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
        store = FakeVectorStore()
        monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
        temp_path = _write_temp(".txt", "alpha beta " * 500)
        assert Path(temp_path).exists()

        document_id = uuid.uuid4()
        users["session"].add(
            DocumentRow(
                id=document_id,
                user_id=users["a"].id,
                filename="notes.txt",
                status=DocumentStatus.PROCESSING,
            )
        )
        users["session"].commit()
        ingest_document(
            str(document_id),
            temp_path,
            "notes.txt",
            str(users["a"].id),
            embeddings_factory=FakeEmbeddings,
            session_factory=_SessionFactory(users["session"]),
        )

        row = users["session"].get(DocumentRow, document_id)
        assert row is not None
        assert row.status == DocumentStatus.READY
        assert row.chunk_count > 0
        assert row.error is None
        assert store.added, "chunks were not added to the store"
        first_chunk = store.added[0][0]
        assert first_chunk.metadata["document_id"] == str(document_id)
        assert first_chunk.metadata["source"] == "notes.txt"
        assert first_chunk.metadata["chunk_index"] == 0
        assert not Path(temp_path).exists(), "temp file was not deleted after ingestion"

    def test_failure_marks_failed_and_still_deletes_temp(self, users, monkeypatch):
        store = FakeVectorStore()
        monkeypatch.setattr(ingest_module, "build_vectorstore", lambda embeddings=None: store)
        temp_path = _write_temp(".txt", "will never load")

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
            embeddings_factory=FakeEmbeddings,
            session_factory=_SessionFactory(users["session"]),
        )

        failed_row = users["session"].get(DocumentRow, row.id)
        assert failed_row.status == DocumentStatus.FAILED
        assert failed_row.chunk_count == 0
        assert "Unsupported file type" in (failed_row.error or "")
        assert not store.added, "nothing should reach the store after loading fails"
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
