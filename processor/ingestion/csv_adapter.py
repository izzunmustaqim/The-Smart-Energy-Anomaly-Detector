"""
Generic CSV data source adapter.

Provides a simple way to load any CSV file with a datetime column
as a DataSource, enabling future extensibility without touching
existing ingestion code.
"""

import logging
from pathlib import Path

import pandas as pd

from processor.ingestion.base import DataSource

logger = logging.getLogger(__name__)


class CSVAdapter(DataSource):
    """
    Load a generic CSV file as a time-indexed DataFrame.

    Parameters
    ----------
    file_path : Path
        Absolute or relative path to the CSV file.
    datetime_col : str
        Name of the column containing timestamps.
    datetime_format : str or None
        strftime format string. If None, pandas infers automatically.
    sep : str
        Column delimiter (default comma).
    """

    def __init__(
        self,
        file_path: Path,
        datetime_col: str = "datetime",
        datetime_format: str | None = None,
        sep: str = ",",
    ) -> None:
        self._file_path = Path(file_path)
        self._datetime_col = datetime_col
        self._datetime_format = datetime_format
        self._sep = sep

    def load(self) -> pd.DataFrame:
        if not self._file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._file_path}")

        logger.info("Loading CSV from %s", self._file_path)

        df = pd.read_csv(self._file_path, sep=self._sep, low_memory=False)

        if self._datetime_col not in df.columns:
            raise KeyError(
                f"Column '{self._datetime_col}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

        df[self._datetime_col] = pd.to_datetime(
            df[self._datetime_col],
            format=self._datetime_format,
        )
        df.set_index(self._datetime_col, inplace=True)
        df.sort_index(inplace=True)

        logger.info("Loaded %s rows × %s columns", len(df), len(df.columns))
        return df
