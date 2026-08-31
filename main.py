"""
NeuroPlay-AI entry point.
Phase 1: Verifies environment setup only. Real CLI/game logic is added in later phases.
"""

from neuroplay.config import settings
from neuroplay.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("NeuroPlay-AI booting up...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Data directory: {settings.data_dir}")
    logger.info("Phase 1 environment verification successful.")


if __name__ == "__main__":
    main()
