"""Tests for preprocessing — sliding window creation and match-level splitting."""

import pandas as pd

from neuroplay.preprocessing.splitter import split_by_match
from neuroplay.preprocessing.windowing import create_windows


def test_create_windows_produces_correct_target_and_window_size():
    df = pd.DataFrame(
        {
            "match_id": [1] * 10,
            "round_number": list(range(1, 11)),
            "player_move": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            "ai_move": [1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
            "model_used": ["test_persona"] * 10,
        }
    )
    windowed = create_windows(df, window_size=3)
    assert len(windowed) == 7  # 10 rounds - 3 window_size = 7 valid windows
    first_row = windowed.iloc[0]
    assert first_row["player_window"] == [0, 1, 2]
    assert first_row["target_move"] == 0


def test_split_by_match_has_no_overlapping_match_ids():
    df = pd.DataFrame(
        {
            "match_id": list(range(1, 21)),
            "model_used": ["persona_a"] * 10 + ["persona_b"] * 10,
        }
    )
    train, val, test = split_by_match(df)
    train_ids = set(train["match_id"])
    val_ids = set(val["match_id"])
    test_ids = set(test["match_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
