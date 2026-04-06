"""
Abstract base class for anomaly detection models.

Defines the contract every model must implement, enabling the
Strategy Pattern for swappable model backends. The detector
orchestrator depends only on this abstraction (DIP).
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class AnomalyModel(ABC):
    """Interface for all anomaly detection models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        """
        Train / fit the model on historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame with datetime index.
        """
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Score each data point for anomalousness.

        Parameters
        ----------
        df : pd.DataFrame
            Same schema as the training data.

        Returns
        -------
        pd.Series
            Anomaly scores in [0, 1] where 1 = most anomalous.
            Index must match the input DataFrame.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artifacts to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model artifacts from disk."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
