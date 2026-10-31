from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

from .config import get_settings


class Base(DeclarativeBase):
    pass


def _build_engine():
    url = get_settings().DATABASE_URL
    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        # SQLite only honours FK ON DELETE CASCADE with this pragma, set per
        # connection so passive_deletes cascades in tests too.
        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    # Neon's free tier suspends idle compute and closes idle connections,
    # so dropped connections must be detected and recycled proactively.
    kwargs: dict = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }
    return create_engine(url, **kwargs)


engine = _build_engine()

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    from . import models

    Base.metadata.create_all(bind=engine)


# Columns added after the first release; create_all only creates missing
# TABLES, never columns, so the startup lifespan applies these by hand.
_DOCUMENT_MIGRATION_COLUMNS: dict[str, str] = {
    # Generic types that are valid on both Postgres and SQLite. The NOT NULL
    # column carries a constant server default, so pre-existing rows backfill
    # as "recursive" with NULL sizes.
    "chunking_strategy": (
        "ALTER TABLE documents ADD COLUMN chunking_strategy "
        "VARCHAR(50) NOT NULL DEFAULT 'recursive'"
    ),
    "chunk_size": "ALTER TABLE documents ADD COLUMN chunk_size INTEGER NULL",
    "chunk_overlap": "ALTER TABLE documents ADD COLUMN chunk_overlap INTEGER NULL",
}


def migrate_document_columns(target_engine=None) -> None:
    """Add missing documents columns; idempotent and never drops data.

    Runs ALTER TABLE ... ADD COLUMN for each column the live table does not
    have yet, inside one transaction. Any failure is logged with a clear
    message and re-raised so the app does not boot half-migrated.
    """
    import logging

    from sqlalchemy import inspect, text

    logger = logging.getLogger("app.db")
    target = target_engine if target_engine is not None else engine
    with target.connect() as connection:
        existing = {column["name"] for column in inspect(connection).get_columns("documents")}
        for column, statement in _DOCUMENT_MIGRATION_COLUMNS.items():
            if column in existing:
                continue
            try:
                connection.execute(text(statement))
            except Exception as error:
                logger.error(
                    "Startup migration failed for documents.%s (%s); the app "
                    "will not start half-migrated.",
                    column,
                    error,
                )
                raise
        connection.commit()
