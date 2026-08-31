"""Phase 1 smoke test — verifies environment and package are correctly configured."""

from neuroplay.config import settings
from neuroplay.constants import BEATS, Move
from neuroplay.logger import get_logger


def test_settings_load():
    assert settings.app_name == "NeuroPlay-AI"
    assert settings.data_dir.name == "data"


def test_logger_returns_configured_logger():
    logger = get_logger("test")
    assert logger.name == "test"
    assert len(logger.handlers) >= 1


def test_game_rules_consistency():
    assert BEATS[Move.ROCK] == Move.SCISSORS
    assert BEATS[Move.PAPER] == Move.ROCK
    assert BEATS[Move.SCISSORS] == Move.PAPER
