"""
Visualization utilities for SHAP explanations — summary plots and
per-prediction waterfall-style bar charts.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from neuroplay.constants import MOVE_NAMES, Move
from neuroplay.logger import get_logger

logger = get_logger(__name__)


def plot_single_explanation(explanation: dict, output_path: Path, top_n: int = 8) -> None:
    """Bar chart of the top-N most influential features for a single prediction."""
    attributions = explanation["feature_attributions"]
    sorted_features = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    features, values = zip(*sorted_features)
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

    plt.figure(figsize=(7, 4))
    plt.barh(features, values, color=colors)
    plt.axvline(x=0, color="black", linewidth=0.8)
    predicted_name = MOVE_NAMES[Move(explanation["predicted_move"])]
    plt.title(f"Feature Attribution — Predicted: {predicted_name}")
    plt.xlabel("SHAP value (impact on prediction)")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved explanation plot to {output_path}")


def plot_global_feature_importance(
    explanations: list[dict], output_path: Path, top_n: int = 10
) -> None:
    """Aggregates mean |SHAP value| across many predictions for global feature importance."""
    all_attrs = pd.DataFrame([e["feature_attributions"] for e in explanations])
    mean_abs_importance = all_attrs.abs().mean().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(7, 5))
    mean_abs_importance.sort_values().plot(kind="barh", color="#2ca02c")
    plt.title("Global Feature Importance (mean |SHAP value|)")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved global feature importance plot to {output_path}")
