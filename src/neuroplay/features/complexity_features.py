"""
Rolling Lempel-Ziv complexity feature — quantifies local sequence
predictability within a sliding window, feeding both modeling (Phase 8-11)
and psychology classification (Phase 21).
"""

import pandas as pd

from neuroplay.eda.complexity import lempel_ziv_complexity


def compute_rolling_lz_complexity(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()
    lz_values = []

    for _, group in df.groupby("match_id"):
        group = group.sort_values("round_number")
        moves = group["player_move"].tolist()
        match_lz = []
        for i in range(len(moves)):
            start = max(0, i - window + 1)
            window_moves = moves[start : i + 1]
            match_lz.append(lempel_ziv_complexity(window_moves))
        lz_values.extend(match_lz)

    df["rolling_lz_complexity_10"] = lz_values
    return df
