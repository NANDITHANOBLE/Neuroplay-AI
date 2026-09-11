"""
Shared dependency injection — loads models ONCE at app startup and provides
them to route handlers, avoiding per-request reload latency.
"""

from functools import lru_cache

from neuroplay.config import settings
from neuroplay.db.session import get_session
from neuroplay.logger import get_logger
from neuroplay.models.ann_model import ANNBaseline

logger = get_logger(__name__)


@lru_cache
def get_ann_model() -> ANNBaseline:
    """Loads the production ANN model + scaler once, cached for the app lifetime."""
    logger.info("Loading ANN model (cached singleton)...")
    ann_dir = settings.models_dir / "ann"
    ann = ANNBaseline(device="cpu")
    ann.load_weights(str(ann_dir / "ann_best.pt"))
    ann.load_scaler(str(ann_dir / "feature_scaler.json"))
    return ann


def get_db_session():
    """FastAPI dependency — yields a DB session, closed after the request."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()
