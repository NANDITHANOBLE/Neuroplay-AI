"""
Initializes the database — creates all tables from ORM models.
Run this once per environment: `python -m neuroplay.db.init_db`
"""

from neuroplay.db.models import Base
from neuroplay.db.session import engine
from neuroplay.logger import get_logger

logger = get_logger(__name__)


def init_db() -> None:
    logger.info("Creating all tables from ORM models...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database initialized successfully.")


if __name__ == "__main__":
    init_db()
