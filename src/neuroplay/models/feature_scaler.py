"""
Feature scaling utilities — standardizes features using training-set statistics
only (mean/std), then applies the same transform consistently to val/test/inference
data. Critical for neural network training stability (Phase 9-11).
"""

import json
from pathlib import Path

import pandas as pd


def fit_scaler(train_df: pd.DataFrame, feature_columns: list[str]) -> dict:
    """Computes per-feature mean and std from the training set only."""
    means = train_df[feature_columns].mean().to_dict()
    stds = train_df[feature_columns].std().replace(0, 1).to_dict()  # avoid div-by-zero
    return {"mean": means, "std": stds, "feature_columns": feature_columns}


def apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    """Applies z-score normalization using previously fitted scaler statistics."""
    df = df.copy()
    for col in scaler["feature_columns"]:
        df[col] = (df[col] - scaler["mean"][col]) / scaler["std"][col]
    return df


def save_scaler(scaler: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(scaler, f, indent=2)


def load_scaler(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
