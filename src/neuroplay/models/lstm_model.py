"""
LSTM sequence model for next-move prediction — exploits raw temporal order
of move history, unlike the ANN's aggregated-feature approach (Phase 9).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from neuroplay.models.base_model import BaseMovePredictor


class LSTMNetwork(nn.Module):
    """Embeds player & AI move sequences, concatenates, feeds through LSTM -> Linear."""

    def __init__(
        self,
        num_moves: int = 3,
        embed_dim: int = 8,
        hidden_dim: int = 32,
        num_classes: int = 3,
    ):
        super().__init__()
        self.player_embedding = nn.Embedding(num_moves, embed_dim)
        self.ai_embedding = nn.Embedding(num_moves, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim * 2, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, player_seq: torch.Tensor, ai_seq: torch.Tensor) -> torch.Tensor:
        player_emb = self.player_embedding(player_seq)  # (batch, seq_len, embed_dim)
        ai_emb = self.ai_embedding(ai_seq)
        combined = torch.cat([player_emb, ai_emb], dim=-1)  # (batch, seq_len, embed_dim*2)
        _, (hidden, _) = self.lstm(combined)
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        return self.fc(last_hidden)


class LSTMBaseline(BaseMovePredictor):
    """Wraps LSTMNetwork to satisfy the BaseMovePredictor interface."""

    name = "lstm_model"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = LSTMNetwork().to(device)

    def fit(self, train_df: pd.DataFrame) -> None:
        pass  # Training happens in train_lstm.py

    def load_weights(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def predict(self, df: pd.DataFrame) -> list[int]:
        self.model.eval()
        player_seq = torch.tensor(np.stack(df["player_window"].to_numpy()), dtype=torch.long).to(
            self.device
        )
        ai_seq = torch.tensor(np.stack(df["ai_window"].to_numpy()), dtype=torch.long).to(
            self.device
        )
        with torch.no_grad():
            logits = self.model(player_seq, ai_seq)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds.tolist()
