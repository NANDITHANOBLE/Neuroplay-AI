"""
Streak-based features — captures win-stay/lose-shift behavioral signal
identified as the dominant human heuristic in Phase 0 research.
"""

import pandas as pd


def compute_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a signed streak_count column per match: positive = consecutive wins,
    negative = consecutive losses, reset to 0 on a draw.
    Expects df sorted by match_id, round_number with a 'result' column.
    """
    df = df.copy()
    streaks = []

    for _, group in df.groupby("match_id"):
        group = group.sort_values("round_number")
        streak = 0
        match_streaks = []
        for result in group["result"]:
            if result == "win":
                streak = streak + 1 if streak >= 0 else 1
            elif result == "loss":
                streak = streak - 1 if streak <= 0 else -1
            else:  # draw
                streak = 0
            match_streaks.append(streak)
        streaks.extend(match_streaks)

    df["streak_count"] = streaks
    return df
