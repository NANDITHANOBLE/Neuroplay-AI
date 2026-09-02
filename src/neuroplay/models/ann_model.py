"""
Feed-forward ANN model for next-move prediction, implementing the shared
BaseMovePredictor interface used by baselines (Phase 8) and future models.
"""

import pandas as pd
import torch
import torch.nn as nn

from neuroplay.models.base_model import BaseMovePredictor
from neuroplay.models.feature_scaler import apply_scaler, load_scaler
from neuroplay.models.torch_dataset import FEATURE_COLUMNS


class ANNNetwork(nn.Module):
    """Simple 2-hidden-layer feed-forward network: 21 -> 64 -> 32 -> 3."""

    def __init__(self, input_dim: int = len(FEATURE_COLUMNS), num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ANNBaseline(BaseMovePredictor):
    """Wraps ANNNetwork to satisfy the BaseMovePredictor interface for evaluation reuse."""

    name = "ann_model"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = ANNNetwork().to(device)
        self.scaler: dict | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        # Training happens in train_ann.py; fit() here is a no-op to satisfy the interface
        # when using a pre-trained model loaded from checkpoint.
        pass

    def load_weights(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def load_scaler(self, path: str) -> None:
        self.scaler = load_scaler(path)

    def predict(self, df: pd.DataFrame) -> list[int]:
        self.model.eval()
        if self.scaler is not None:
            df = apply_scaler(df, self.scaler)
        features = torch.tensor(df[FEATURE_COLUMNS].to_numpy(), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(features)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds.tolist()
