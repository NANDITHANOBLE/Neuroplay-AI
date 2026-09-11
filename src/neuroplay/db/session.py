"""
Database session and engine setup with connection pooling for production use.
Uses SQLite by default (from settings.database_url); swaps to PostgreSQL
transparently by changing the DATABASE_URL env var only.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from neuroplay.config import settings

is_sqlite = "sqlite" in settings.database_url

if is_sqlite:
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
else:
    engine = create_engine(
        settings.database_url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session() -> Session:
    """Yields a DB session; use as a context manager or FastAPI dependency."""
    return SessionLocal()
