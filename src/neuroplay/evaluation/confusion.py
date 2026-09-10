"""
Confusion matrix computation and visualization for model predictions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from neuroplay.constants import MOVE_NAMES, Move
from neuroplay.logger import get_logger

logger = get_logger(__name__)


def compute_confusion_matrix(actuals: list[int], predictions: list[int]) -> np.ndarray:
    return confusion_matrix(actuals, predictions, labels=[0, 1, 2])


def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [MOVE_NAMES[Move(i)] for i in range(3)]

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.close()
    logger.info(f"Saved confusion_matrix_{model_name}.png")
