"""
Hybrid dataset — merges raw windowed move sequences (Phase 5) with
Phase 7's engineered features, joined causally on the PRIOR round
(round_number - 1) to avoid leaking the current round's outcome.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from neuroplay.models.torch_dataset import FEATURE_COLUMNS


def merge_features(windowed_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins windowed sequence data with engineered features from the round
    BEFORE the target round (causally correct — avoids leakage).
    """
    feat = features_df[["match_id", "round_number"] + FEATURE_COLUMNS].copy()
    feat = feat.rename(columns={"round_number": "feature_round"})

    merged = windowed_df.copy()
    merged["feature_round"] = merged["round_number"] - 1

    merged = merged.merge(feat, on=["match_id", "feature_round"], how="left")
    merged[FEATURE_COLUMNS] = merged[FEATURE_COLUMNS].fillna(0)
    return merged


class HybridDataset(Dataset):
    """Wraps merged sequence + engineered-feature data for the hybrid Transformer."""

    def __init__(self, df: pd.DataFrame):
        self.player_windows = torch.tensor(
            np.stack(df["player_window"].to_numpy()), dtype=torch.long
        )
        self.ai_windows = torch.tensor(np.stack(df["ai_window"].to_numpy()), dtype=torch.long)
        self.features = torch.tensor(df[FEATURE_COLUMNS].to_numpy(), dtype=torch.float32)
        self.targets = torch.tensor(df["target_move"].to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return (
            self.player_windows[idx],
            self.ai_windows[idx],
            self.features[idx],
            self.targets[idx],
        )
