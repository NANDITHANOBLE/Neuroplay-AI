"""
Visualization utilities for exploratory data analysis.
Generates and saves charts to assets/eda/ for README and dashboard use.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from neuroplay.config import settings
from neuroplay.constants import MOVE_NAMES, Move
from neuroplay.logger import get_logger

logger = get_logger(__name__)

ASSETS_DIR = settings.base_dir / "assets" / "eda"


def _ensure_assets_dir() -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    return ASSETS_DIR


def plot_move_distribution(df: pd.DataFrame) -> None:
    """Bar chart of overall player move frequency."""
    assets_dir = _ensure_assets_dir()
    counts = df["player_move"].value_counts().sort_index()
    labels = [MOVE_NAMES[Move(i)] for i in counts.index]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=counts.values, hue=labels, legend=False)
    plt.title("Overall Player Move Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(assets_dir / "move_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved move_distribution.png")


def plot_win_rate_by_persona(df: pd.DataFrame) -> None:
    """Bar chart comparing win/loss/draw rate across personas."""
    assets_dir = _ensure_assets_dir()
    result_counts = df.groupby(["model_used", "result"]).size().unstack(fill_value=0)
    result_rates = result_counts.div(result_counts.sum(axis=1), axis=0)

    result_rates.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="viridis")
    plt.title("Win/Loss/Draw Rate by Persona")
    plt.ylabel("Proportion")
    plt.xlabel("Persona")
    plt.legend(title="Result")
    plt.tight_layout()
    plt.savefig(assets_dir / "win_rate_by_persona.png", dpi=150)
    plt.close()
    logger.info("Saved win_rate_by_persona.png")


def plot_transition_heatmap(df: pd.DataFrame, persona: str) -> None:
    """Heatmap of P(next_move | current_move) for a given persona."""
    assets_dir = _ensure_assets_dir()
    persona_df = df[df["model_used"] == persona].sort_values(["match_id", "round_number"])

    transitions = pd.DataFrame(0, index=range(3), columns=range(3))
    for _, group in persona_df.groupby("match_id"):
        moves = group["player_move"].to_numpy()
        for i in range(len(moves) - 1):
            transitions.loc[moves[i], moves[i + 1]] += 1

    transition_probs = transitions.div(transitions.sum(axis=1), axis=0).fillna(0)
    labels = [MOVE_NAMES[Move(i)] for i in range(3)]

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        transition_probs,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
    )
    plt.title(f"Move Transition Probabilities — {persona}")
    plt.xlabel("Next Move")
    plt.ylabel("Current Move")
    plt.tight_layout()
    plt.savefig(assets_dir / f"transition_heatmap_{persona}.png", dpi=150)
    plt.close()
    logger.info(f"Saved transition_heatmap_{persona}.png")


def plot_drift_visualization(df: pd.DataFrame, match_id: int, window: int = 10) -> None:
    """
    Rolling win-rate plot for a single drifting-bot match,
    to visually confirm the strategy switch.
    """
    assets_dir = _ensure_assets_dir()
    match_df = df[df["match_id"] == match_id].sort_values("round_number")

    match_df = match_df.copy()
    match_df["is_win"] = (match_df["result"] == "win").astype(int)
    match_df["rolling_win_rate"] = match_df["is_win"].rolling(window=window).mean()

    plt.figure(figsize=(8, 4))
    plt.plot(match_df["round_number"], match_df["rolling_win_rate"])
    plt.axvline(x=50, color="red", linestyle="--", label="Expected drift point")
    plt.title(f"Rolling Win Rate — Drifting Bot (match_id={match_id})")
    plt.xlabel("Round Number")
    plt.ylabel(f"Rolling Win Rate (window={window})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(assets_dir / "drift_visualization.png", dpi=150)
    plt.close()
    logger.info("Saved drift_visualization.png")
