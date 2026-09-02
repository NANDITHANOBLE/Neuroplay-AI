"""
Rolling window features — win rate, move frequency, and reaction time
statistics computed over recent rounds within each match.
"""

import pandas as pd

from neuroplay.constants import Move


def compute_rolling_win_rate(df: pd.DataFrame, windows: tuple[int, ...] = (5, 20)) -> pd.DataFrame:
    df = df.copy()
    df["is_win"] = (df["result"] == "win").astype(int)

    for w in windows:
        col = f"rolling_win_rate_{w}"
        df[col] = df.groupby("match_id")["is_win"].transform(
            lambda s: s.rolling(window=w, min_periods=1).mean()
        )
    return df


def compute_move_frequency(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()
    for move in Move:
        indicator_col = f"_is_move_{move.value}"
        freq_col = f"move_freq_{move.name.lower()}"
        df[indicator_col] = (df["player_move"] == move.value).astype(int)
        df[freq_col] = df.groupby("match_id")[indicator_col].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        df.drop(columns=[indicator_col], inplace=True)
    return df


def compute_reaction_time_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["reaction_time_mean_5"] = df.groupby("match_id")["reaction_time_ms"].transform(
        lambda s: s.rolling(window=window, min_periods=1).mean()
    )
    df["reaction_time_std_5"] = (
        df.groupby("match_id")["reaction_time_ms"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).std())
        .fillna(0)
    )
    return df


def compute_last_move_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes the PREVIOUS round's move (shifted by 1 within each match).
    First round of each match has no previous move -> all flags are 0.
    """
    df = df.copy()
    df["_prev_move"] = df.groupby("match_id")["player_move"].shift(1)
    for move in Move:
        df[f"last_move_is_{move.name.lower()}"] = (df["_prev_move"] == move.value).astype(int)
    df.drop(columns=["_prev_move"], inplace=True)
    return df


def compute_round_position(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    match_lengths = df.groupby("match_id")["round_number"].transform("max")
    df["round_number_normalized"] = df["round_number"] / match_lengths
    return df
