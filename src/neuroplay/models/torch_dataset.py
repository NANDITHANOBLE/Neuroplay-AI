"""
PyTorch Dataset wrapper for the engineered feature set (Phase 7 output),
used by ANN (Phase 9), LSTM (Phase 10), and Transformer (Phase 11).
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURE_COLUMNS = [
    "streak_count",
    "rolling_win_rate_5",
    "rolling_win_rate_20",
    "move_freq_rock",
    "move_freq_paper",
    "move_freq_scissors",
    "reaction_time_mean_5",
    "reaction_time_std_5",
    "last_move_is_rock",
    "last_move_is_paper",
    "last_move_is_scissors",
    "round_number_normalized",
    "rolling_lz_complexity_10",
]

TARGET_COLUMN = "player_move"  # next-move label; for the ANN we predict the *current* row's
# move using prior rounds' rolling stats (features are computed causally per Phase 7)


class MoveDataset(Dataset):
    """Wraps a features DataFrame for PyTorch training. Predicts player_move from features."""

    def __init__(self, df: pd.DataFrame, feature_columns: list[str] = None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.features = torch.tensor(df[self.feature_columns].to_numpy(), dtype=torch.float32)
        self.targets = torch.tensor(df[TARGET_COLUMN].to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]
