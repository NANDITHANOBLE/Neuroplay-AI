"""
Creates sliding-window sequences from per-match move data.
Each window: last N moves -> next player move (the prediction target).
"""

import pandas as pd

from neuroplay.logger import get_logger

logger = get_logger(__name__)

WINDOW_SIZE = 5


def create_windows(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """
    For each match_id, builds sliding windows of `window_size` past moves
    (player_move, ai_move pairs) as input, with the next player_move as target.
    Returns a new DataFrame with one row per valid window.
    """
    records = []

    for match_id, group in df.groupby("match_id"):
        group = group.sort_values("round_number").reset_index(drop=True)
        player_moves = group["player_move"].to_numpy()
        ai_moves = group["ai_move"].to_numpy()
        model_used = group["model_used"].iloc[0]

        for i in range(window_size, len(group)):
            player_window = player_moves[i - window_size : i]
            ai_window = ai_moves[i - window_size : i]
            target = player_moves[i]

            records.append(
                {
                    "match_id": match_id,
                    "round_number": group["round_number"].iloc[i],
                    "player_window": player_window.tolist(),
                    "ai_window": ai_window.tolist(),
                    "target_move": int(target),
                    "model_used": model_used,
                }
            )

    windowed_df = pd.DataFrame(records)
    logger.info(
        f"Created {len(windowed_df)} windowed sequences (window_size={window_size}) "
        f"from {df['match_id'].nunique()} matches."
    )
    return windowed_df
