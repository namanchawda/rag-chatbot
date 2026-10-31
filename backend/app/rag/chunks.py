"""Read-only access to the langchain_postgres chunk tables.

The real schema is defined by the installed langchain_postgres 0.0.19
(``_get_embedding_collection_store`` in
venv/Lib/site-packages/langchain_postgres/vectorstores.py):

    langchain_pg_collection (CollectionStore):
        uuid        UUID, primary key
        name        String, unique
        cmetadata   JSON

    langchain_pg_embedding (EmbeddingStore):
        id            String, primary key ("<document_id>:<chunk_index>")
        collection_id UUID, FK -> langchain_pg_collection.uuid (CASCADE)
        embedding     pgvector.sqlalchemy.Vector(vector_dimension)
        document      String, nullable - the chunk text
        cmetadata     JSONB, nullable - holds user_id (str), document_id (str),
                      source (filename str) and chunk_index (int), written by
                      app/rag/ingest.py:_annotate

This module mirrors those tables with the exact column names but dialect-safe
types (generic JSON instead of JSONB, no Vector column) so identical queries
run against the production Postgres tables and the SQLite mirror used in
tests. Deployment never issues DDL from here: langchain_postgres owns table
creation, and a missing table simply means an empty result.

Ingestion writes user_id, document_id and source into the cmetadata JSON
column, so every ownership filter below IS the SQL query itself: another
user's rows are never fetched, in counts, lists, or stats.
"""

from sqlalchemy import (
    Column,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    and_,
    cast,
    func,
    select,
)

from .ingest import COLLECTION_NAME

EMBEDDING_TABLE_NAME = "langchain_pg_embedding"
COLLECTION_TABLE_NAME = "langchain_pg_collection"

# Own metadata, NOT the app Base: in production langchain_postgres owns these
# tables' DDL, while tests build the SQLite mirror from this metadata.
chunks_metadata = MetaData()

collection_table = Table(
    COLLECTION_TABLE_NAME,
    chunks_metadata,
    Column("uuid", String, primary_key=True),
    Column("name", String(255), nullable=False, unique=True),
    Column("cmetadata", JSON),
)

# Mirrors EmbeddingStore minus the `embedding` Vector column: that column is
# never referenced in any projection, so vectors can never leave the database.
embedding_table = Table(
    EMBEDDING_TABLE_NAME,
    chunks_metadata,
    Column("id", String, primary_key=True),
    Column("collection_id", String),
    Column("document", String),
    Column("cmetadata", JSON),
)

_LIKE_ESCAPE_CHARACTER = chr(92)


def _metadata_text(key: str):
    """Compile-safe text extraction from the cmetadata JSON column.

    On Postgres this renders ``cmetadata ->> 'key'`` (correct for both the
    jsonb column langchain_postgres creates and our json mirror); on SQLite
    it renders json_extract. Every value is a bound parameter.
    """
    return embedding_table.c.cmetadata[key].as_string()


def _chunk_index_order():
    """Numeric chunk_index ordering; an absent index sorts as 0."""
    return cast(func.coalesce(_metadata_text("chunk_index"), "0"), Integer).asc()


def _source_order():
    return func.coalesce(_metadata_text("source"), "")


def _contains_pattern(q: str) -> str:
    """Substring pattern with %, _ and backslash escaped to match literally."""
    escaped = (
        q.replace(_LIKE_ESCAPE_CHARACTER, _LIKE_ESCAPE_CHARACTER * 2)
        .replace("%", _LIKE_ESCAPE_CHARACTER + "%")
        .replace("_", _LIKE_ESCAPE_CHARACTER + "_")
    )
    return f"%{escaped}%"


def _collection_join():
    return and_(
        embedding_table.c.collection_id == collection_table.c.uuid,
        collection_table.c.name == COLLECTION_NAME,
    )


def _tables_present(connection) -> bool:
    return bool(
        connection.dialect.has_table(connection, COLLECTION_TABLE_NAME)
        and connection.dialect.has_table(connection, EMBEDDING_TABLE_NAME)
    )


def _base_filters(user_id: str, document_id: str | None, q: str | None) -> list:
    filters = [_metadata_text("user_id") == user_id]
    if document_id is not None:
        filters.append(_metadata_text("document_id") == document_id)
    if q is not None:
        filters.append(
            embedding_table.c.document.ilike(
                _contains_pattern(q), escape=_LIKE_ESCAPE_CHARACTER
            )
        )
    return filters


def _row_to_item(row) -> dict:
    text = row.document or ""
    metadata = row.cmetadata if isinstance(row.cmetadata, dict) else {}
    char_count = len(text)
    return {
        "id": str(row.id),
        "document_id": str(metadata.get("document_id", "")),
        "source": str(metadata.get("source", "")),
        "chunk_index": int(metadata.get("chunk_index", 0) or 0),
        "text": text,
        "char_count": char_count,
        "token_estimate": char_count // 4,
    }


def _select_columns():
    return select(
        embedding_table.c.id,
        embedding_table.c.document,
        embedding_table.c.cmetadata,
    ).join(collection_table, _collection_join())


def list_chunks(
    user_id: str,
    *,
    document_id: str | None = None,
    q: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """The caller's chunks, ordered by (source filename, chunk_index).

    Returns (items, total); items are the offset/limit page. The projection
    never includes the embedding column, so vectors can't leave the database.
    """
    from ..db import engine

    with engine.connect() as connection:
        if not _tables_present(connection):
            return [], 0
        filters = _base_filters(user_id, document_id, q)
        total = connection.scalar(
            select(func.count())
            .select_from(embedding_table)
            .join(collection_table, _collection_join())
            .where(*filters)
        )
        rows = connection.execute(
            _select_columns()
            .where(*filters)
            .order_by(_source_order().asc(), _chunk_index_order())
            .limit(limit)
            .offset(offset)
        ).all()
    return [_row_to_item(row) for row in rows], int(total or 0)


def get_chunk(user_id: str, chunk_id: str) -> dict | None:
    """One chunk by exact id; the user filter is part of the SQL, so a
    foreign id is an ordinary miss (None -> 404)."""
    from ..db import engine

    with engine.connect() as connection:
        if not _tables_present(connection):
            return None
        row = connection.execute(
            _select_columns()
            .where(
                embedding_table.c.id == chunk_id,
                _metadata_text("user_id") == user_id,
            )
            .limit(1)
        ).first()
    if row is None:
        return None
    return _row_to_item(row)


def chunk_stats(user_id: str, ready_document_ids: list[str]) -> dict:
    """{documents, chunks, characters} across READY documents.

    The caller passes the owner's READY document ids (from the documents
    table, filtered by user and status inside SQL there); here the chunk
    rows are filtered by user_id inside SQL as well and matched to exactly
    those document ids.
    """
    from ..db import engine

    if not ready_document_ids:
        return {"documents": 0, "chunks": 0, "characters": 0}
    with engine.connect() as connection:
        if not _tables_present(connection):
            return {"documents": 0, "chunks": 0, "characters": 0}
        row = connection.execute(
            select(
                func.count(),
                func.coalesce(func.sum(func.length(embedding_table.c.document)), 0),
            )
            .select_from(embedding_table)
            .join(collection_table, _collection_join())
            .where(
                _metadata_text("user_id") == user_id,
                _metadata_text("document_id").in_(ready_document_ids),
            )
        ).one()
    return {
        "documents": len(ready_document_ids),
        "chunks": int(row[0] or 0),
        "characters": int(row[1] or 0),
    }
