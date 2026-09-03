"""
Training script for the LSTM sequence model.
Run: python -m neuroplay.models.train_lstm
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuroplay.config import settings
from neuroplay.logger import get_logger
from neuroplay.models.lstm_dataset import SequenceDataset
from neuroplay.models.lstm_model import LSTMBaseline, LSTMNetwork

logger = get_logger(__name__)

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def main() -> None:
    processed_dir = settings.data_dir / "processed"

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_loader = DataLoader(SequenceDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_df), batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    model = LSTMNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    models_dir = settings.models_dir / "lstm"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "lstm_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for player_seq, ai_seq, targets in train_loader:
            player_seq, ai_seq, targets = (
                player_seq.to(device),
                ai_seq.to(device),
                targets.to(device),
            )
            optimizer.zero_grad()
            outputs = model(player_seq, ai_seq)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for player_seq, ai_seq, targets in val_loader:
                player_seq, ai_seq, targets = (
                    player_seq.to(device),
                    ai_seq.to(device),
                    targets.to(device),
                )
                outputs = model(player_seq, ai_seq)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        val_acc = correct / total if total > 0 else 0.0

        logger.info(
            f"Epoch {epoch}/{EPOCHS} | Train Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"  -> New best model saved (val_acc={val_acc:.4f})")

    logger.info(f"✅ Training complete. Best val accuracy: {best_val_acc:.4f}")

    lstm = LSTMBaseline(device=device)
    lstm.load_weights(str(checkpoint_path))
    predictions = lstm.predict(test_df)
    test_df = test_df.copy()
    test_df["prediction"] = predictions
    test_acc = (test_df["prediction"] == test_df["target_move"]).mean()
    logger.info(f"✅ Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
