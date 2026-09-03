"""
Training script for the hybrid Transformer model.
Run: python -m neuroplay.models.train_transformer
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuroplay.config import settings
from neuroplay.logger import get_logger
from neuroplay.models.feature_scaler import apply_scaler, fit_scaler, save_scaler
from neuroplay.models.hybrid_dataset import HybridDataset, merge_features
from neuroplay.models.torch_dataset import FEATURE_COLUMNS
from neuroplay.models.transformer_model import TransformerNetwork

logger = get_logger(__name__)

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def main() -> None:
    processed_dir = settings.data_dir / "processed"

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    features_df = pd.read_parquet(processed_dir / "features_full.parquet")

    logger.info("Merging windowed sequences with engineered features (causal join)...")
    train_merged = merge_features(train_df, features_df)
    val_merged = merge_features(val_df, features_df)
    test_merged = merge_features(test_df, features_df)

    logger.info(f"Train: {len(train_merged)} | Val: {len(val_merged)} | Test: {len(test_merged)}")

    scaler = fit_scaler(train_merged, FEATURE_COLUMNS)
    train_merged = apply_scaler(train_merged, scaler)
    val_merged = apply_scaler(val_merged, scaler)

    models_dir = settings.models_dir / "transformer"
    models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = models_dir / "feature_scaler.json"
    save_scaler(scaler, scaler_path)
    logger.info(f"Fitted and saved feature scaler to {scaler_path}")

    train_loader = DataLoader(HybridDataset(train_merged), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(HybridDataset(val_merged), batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    model = TransformerNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    checkpoint_path = models_dir / "transformer_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for player_seq, ai_seq, features, targets in train_loader:
            player_seq, ai_seq, features, targets = (
                player_seq.to(device),
                ai_seq.to(device),
                features.to(device),
                targets.to(device),
            )
            optimizer.zero_grad()
            outputs = model(player_seq, ai_seq, features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for player_seq, ai_seq, features, targets in val_loader:
                player_seq, ai_seq, features, targets = (
                    player_seq.to(device),
                    ai_seq.to(device),
                    features.to(device),
                    targets.to(device),
                )
                outputs = model(player_seq, ai_seq, features)
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

    from neuroplay.models.transformer_model import TransformerBaseline

    transformer = TransformerBaseline(device=device)
    transformer.load_weights(str(checkpoint_path))
    transformer.load_scaler(str(scaler_path))

    predictions = transformer.predict(test_df, features_df)
    test_eval_df = merge_features(test_df, features_df).copy()
    test_eval_df["prediction"] = predictions
    test_acc = (test_eval_df["prediction"] == test_eval_df["target_move"]).mean()
    logger.info(f"✅ Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
