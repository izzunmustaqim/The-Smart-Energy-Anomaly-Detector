"""Tests for the data ingestion layer."""

import pandas as pd
import pytest

from processor.ingestion.base import DataSource
from processor.ingestion.csv_adapter import CSVAdapter


class TestDataSourceInterface:
    """Test the abstract DataSource contract."""

    def test_cannot_instantiate_abstract_class(self):
        """DataSource is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataSource()

    def test_subclass_must_implement_load(self):
        """A subclass without load() should fail."""

        class BrokenAdapter(DataSource):
            pass

        with pytest.raises(TypeError):
            BrokenAdapter()


class TestCSVAdapter:
    """Test the generic CSV adapter."""

    def test_load_valid_csv(self, tmp_path):
        """Should load a well-formed CSV into a DatetimeIndex DataFrame."""
        csv_content = (
            "datetime,value,other\n"
            "2024-01-01 00:00:00,1.5,10\n"
            "2024-01-01 01:00:00,2.0,20\n"
            "2024-01-01 02:00:00,1.8,15\n"
        )
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text(csv_content)

        adapter = CSVAdapter(file_path=csv_file, datetime_col="datetime")
        df = adapter.load()

        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 3
        assert "value" in df.columns
        assert "other" in df.columns

    def test_load_missing_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        adapter = CSVAdapter(file_path=tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            adapter.load()

    def test_load_missing_datetime_col_raises(self, tmp_path):
        """Should raise KeyError if datetime column doesn't exist."""
        csv_file = tmp_path / "no_dt.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4\n")

        adapter = CSVAdapter(file_path=csv_file, datetime_col="timestamp")
        with pytest.raises(KeyError, match="timestamp"):
            adapter.load()
