"""
Database session and engine setup.
Uses SQLite by default (from settings.database_url); swaps to PostgreSQL
transparently in Phase 18 by changing the DATABASE_URL env var only.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from neuroplay.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session() -> Session:
    """Yields a DB session; use as a context manager or FastAPI dependency (Phase 17)."""
    return SessionLocal()
