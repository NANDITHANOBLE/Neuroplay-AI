"""Tests for feature engineering — streak calculation and rolling statistics."""

import pandas as pd

from neuroplay.features.rolling_features import compute_rolling_win_rate
from neuroplay.features.streak_features import compute_streak_features


def _make_match_df(results: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [1] * len(results),
            "round_number": list(range(1, len(results) + 1)),
            "result": results,
        }
    )


def test_streak_count_increments_on_consecutive_wins():
    df = _make_match_df(["win", "win", "win"])
    result = compute_streak_features(df)
    assert result["streak_count"].tolist() == [1, 2, 3]


def test_streak_count_resets_on_draw():
    df = _make_match_df(["win", "win", "draw", "loss"])
    result = compute_streak_features(df)
    assert result["streak_count"].tolist() == [1, 2, 0, -1]


def test_streak_count_goes_negative_on_losses():
    df = _make_match_df(["loss", "loss"])
    result = compute_streak_features(df)
    assert result["streak_count"].tolist() == [-1, -2]


def test_rolling_win_rate_matches_manual_calculation():
    df = _make_match_df(["win", "loss", "win", "win", "loss"])
    result = compute_rolling_win_rate(df, windows=(5,))
    # Cumulative win rate at each step with min_periods=1
    expected = [1 / 1, 1 / 2, 2 / 3, 3 / 4, 3 / 5]
    assert result["rolling_win_rate_5"].round(4).tolist() == [round(e, 4) for e in expected]
