"""
Training script for the ANN baseline model.
Run: python -m neuroplay.models.train_ann
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuroplay.config import settings
from neuroplay.logger import get_logger
from neuroplay.models.ann_model import ANNBaseline, ANNNetwork
from neuroplay.models.feature_scaler import apply_scaler, fit_scaler, save_scaler
from neuroplay.models.torch_dataset import FEATURE_COLUMNS, MoveDataset

logger = get_logger(__name__)

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def train_test_split_features(df: pd.DataFrame, match_ids: set) -> pd.DataFrame:
    return df[df["match_id"].isin(match_ids)].reset_index(drop=True)


def main() -> None:
    processed_dir = settings.data_dir / "processed"
    features_df = pd.read_parquet(processed_dir / "features_full.parquet")

    # Reuse the same match-level split as Phase 5 by reading match_ids from train/val/test
    train_ids = set(pd.read_parquet(processed_dir / "train.parquet")["match_id"].unique())
    val_ids = set(pd.read_parquet(processed_dir / "val.parquet")["match_id"].unique())
    test_ids = set(pd.read_parquet(processed_dir / "test.parquet")["match_id"].unique())

    train_df = train_test_split_features(features_df, train_ids)
    val_df = train_test_split_features(features_df, val_ids)
    test_df = train_test_split_features(features_df, test_ids)

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Fit feature scaler on TRAIN ONLY, then apply to train/val for training.
    # test_df is left UNSCALED — ANNBaseline.predict() applies scaling internally.
    scaler = fit_scaler(train_df, FEATURE_COLUMNS)
    train_df = apply_scaler(train_df, scaler)
    val_df = apply_scaler(val_df, scaler)

    models_dir = settings.models_dir / "ann"
    models_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = models_dir / "feature_scaler.json"
    save_scaler(scaler, scaler_path)
    logger.info(f"Fitted and saved feature scaler to {scaler_path}")

    train_loader = DataLoader(MoveDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MoveDataset(val_df), batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    model = ANNNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    checkpoint_path = models_dir / "ann_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
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

    # Final test-set evaluation
    ann = ANNBaseline(device=device)
    ann.load_weights(str(checkpoint_path))
    ann.load_scaler(str(scaler_path))

    test_predictions = ann.predict(test_df)
    test_df = test_df.copy()
    test_df["prediction"] = test_predictions
    test_acc = (test_df["prediction"] == test_df["player_move"]).mean()
    logger.info(f"✅ Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
