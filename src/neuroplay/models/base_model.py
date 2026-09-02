"""
Abstract base interface that all NeuroPlay-AI models (baseline through Transformer)
implement, ensuring a consistent fit/predict contract across Phases 8-11.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseMovePredictor(ABC):
    """Common interface: fit on training data, predict next move for given windows."""

    name: str = "base"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Trains/fits the model using the windowed training DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> list[int]:
        """Returns predicted next move (0/1/2) for each row in df."""
        raise NotImplementedError
