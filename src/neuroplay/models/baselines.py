"""
Baseline predictors: Random, Majority-class, and Markov Chain (order 1 & 2).
These establish the benchmark that ANN/LSTM/Transformer (Phase 9-11) must beat.
Grounded in Phase 0 research: Markov chains over player move history are the
classical, well-documented approach to exploiting human RPS predictability.
"""

import random
from collections import defaultdict

import pandas as pd

from neuroplay.constants import Move
from neuroplay.models.base_model import BaseMovePredictor


class RandomBaseline(BaseMovePredictor):
    """Predicts uniformly at random — the theoretical floor (~33% accuracy)."""

    name = "random_baseline"

    def fit(self, train_df: pd.DataFrame) -> None:
        pass  # no fitting needed

    def predict(self, df: pd.DataFrame) -> list[int]:
        return [random.choice(list(Move)).value for _ in range(len(df))]


class MajorityBaseline(BaseMovePredictor):
    """Always predicts the most frequent move seen in training data."""

    name = "majority_baseline"

    def __init__(self) -> None:
        self.majority_move: int = 0

    def fit(self, train_df: pd.DataFrame) -> None:
        self.majority_move = int(train_df["target_move"].value_counts().idxmax())

    def predict(self, df: pd.DataFrame) -> list[int]:
        return [self.majority_move for _ in range(len(df))]


class MarkovOrder1Baseline(BaseMovePredictor):
    """
    Learns P(next_move | last_move) from training data.
    Predicts the most likely next move given only the most recent move.
    """

    name = "markov_order1_baseline"

    def __init__(self) -> None:
        self.transition_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def fit(self, train_df: pd.DataFrame) -> None:
        for _, row in train_df.iterrows():
            last_move = row["player_window"][-1]
            target = row["target_move"]
            self.transition_counts[last_move][target] += 1

    def predict(self, df: pd.DataFrame) -> list[int]:
        predictions = []
        for _, row in df.iterrows():
            last_move = row["player_window"][-1]
            counts = self.transition_counts.get(last_move)
            if counts:
                predictions.append(max(counts, key=counts.get))
            else:
                predictions.append(random.choice(list(Move)).value)
        return predictions


class MarkovOrder2Baseline(BaseMovePredictor):
    """
    Learns P(next_move | last_2_moves) from training data.
    The strongest classical baseline — matches the approach used by the
    NYT/Afiniti RPS bot referenced in Phase 0 research.
    """

    name = "markov_order2_baseline"

    def __init__(self) -> None:
        self.transition_counts: dict[tuple[int, int], dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        for _, row in train_df.iterrows():
            last_two = tuple(row["player_window"][-2:])
            target = row["target_move"]
            self.transition_counts[last_two][target] += 1

    def predict(self, df: pd.DataFrame) -> list[int]:
        predictions = []
        for _, row in df.iterrows():
            last_two = tuple(row["player_window"][-2:])
            counts = self.transition_counts.get(last_two)
            if counts:
                predictions.append(max(counts, key=counts.get))
            else:
                predictions.append(random.choice(list(Move)).value)
        return predictions
