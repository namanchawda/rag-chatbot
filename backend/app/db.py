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
