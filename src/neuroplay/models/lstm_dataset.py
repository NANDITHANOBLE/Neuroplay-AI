"""
PyTorch Dataset for LSTM/Transformer models — wraps windowed sequence data
(player_window, ai_window arrays) rather than aggregated features.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Wraps windowed move sequences for sequence models (LSTM, Transformer)."""

    def __init__(self, df: pd.DataFrame):
        self.player_windows = torch.tensor(
            np.stack(df["player_window"].to_numpy()), dtype=torch.long
        )
        self.ai_windows = torch.tensor(np.stack(df["ai_window"].to_numpy()), dtype=torch.long)
        self.targets = torch.tensor(df["target_move"].to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.player_windows[idx], self.ai_windows[idx], self.targets[idx]
