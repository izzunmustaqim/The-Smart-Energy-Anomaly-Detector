"""
Abstract base class for data source adapters.

Every data ingestion adapter must implement the `load` method,
returning a raw pandas DataFrame that downstream preprocessing
can consume. The adapter pattern (OCP) lets us add new data
sources without modifying existing code.
"""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Contract that every data ingestion adapter must satisfy."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load raw data from the source.

        Returns
        -------
        pd.DataFrame
            A DataFrame with at minimum a datetime-like column and
            one or more power-measurement columns.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
