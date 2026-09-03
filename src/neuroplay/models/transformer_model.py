"""
Hybrid Transformer model — self-attention over raw move sequences, fused
with Phase 7's engineered features, aiming to combine the strengths of both
approaches (Phase 9's ANN vs. Phase 10's LSTM).
"""

import pandas as pd
import torch
import torch.nn as nn

from neuroplay.models.base_model import BaseMovePredictor
from neuroplay.models.feature_scaler import apply_scaler, load_scaler
from neuroplay.models.hybrid_dataset import merge_features
from neuroplay.models.torch_dataset import FEATURE_COLUMNS

WINDOW_SIZE = 5


class TransformerNetwork(nn.Module):
    """
    Embeds player/AI move sequences, adds learnable positional encoding,
    runs through a Transformer encoder, pools, then fuses with engineered
    features before the final classification head.
    """

    def __init__(
        self,
        num_moves: int = 3,
        embed_dim: int = 8,
        num_heads: int = 2,
        num_layers: int = 2,
        num_features: int = len(FEATURE_COLUMNS),
        num_classes: int = 3,
        seq_len: int = WINDOW_SIZE,
    ):
        super().__init__()
        combined_dim = embed_dim * 2

        self.player_embedding = nn.Embedding(num_moves, embed_dim)
        self.ai_embedding = nn.Embedding(num_moves, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, combined_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=combined_dim,
            nhead=num_heads,
            dim_feedforward=combined_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.feature_proj = nn.Linear(num_features, 16)

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(
        self, player_seq: torch.Tensor, ai_seq: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        player_emb = self.player_embedding(player_seq)
        ai_emb = self.ai_embedding(ai_seq)
        combined = torch.cat([player_emb, ai_emb], dim=-1)
        combined = combined + self.positional_encoding

        encoded = self.transformer_encoder(combined)
        pooled = encoded.mean(dim=1)  # mean-pool over sequence length

        feat_out = torch.relu(self.feature_proj(features))
        fused = torch.cat([pooled, feat_out], dim=-1)

        return self.classifier(fused)


class TransformerBaseline(BaseMovePredictor):
    """Wraps TransformerNetwork to satisfy the BaseMovePredictor interface."""

    name = "transformer_model"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = TransformerNetwork().to(device)
        self.scaler: dict | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        pass  # Training happens in train_transformer.py

    def load_weights(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def load_scaler(self, path: str) -> None:
        self.scaler = load_scaler(path)

    def predict(self, windowed_df: pd.DataFrame, features_df: pd.DataFrame) -> list[int]:
        self.model.eval()
        merged = merge_features(windowed_df, features_df)
        if self.scaler is not None:
            merged = apply_scaler(merged, self.scaler)

        player_seq = torch.tensor(
            merged["player_window"].apply(list).tolist(), dtype=torch.long
        ).to(self.device)
        ai_seq = torch.tensor(merged["ai_window"].apply(list).tolist(), dtype=torch.long).to(
            self.device
        )
        features = torch.tensor(merged[FEATURE_COLUMNS].to_numpy(), dtype=torch.float32).to(
            self.device
        )

        with torch.no_grad():
            logits = self.model(player_seq, ai_seq, features)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds.tolist()
