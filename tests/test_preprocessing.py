"""Tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from processor.preprocessing.cleaner import clean, resample
from processor.preprocessing.features import (
    add_rate_of_change,
    add_rolling_features,
    add_time_features,
    engineer_features,
)


@pytest.fixture
def sample_minute_data() -> pd.DataFrame:
    """Create sample minute-level energy data resembling UCI format."""
    rng = pd.date_range("2024-01-01", periods=1440, freq="min")  # 1 day
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "Global_active_power": np.random.uniform(0.2, 5.0, len(rng)),
            "Global_reactive_power": np.random.uniform(0.0, 1.0, len(rng)),
            "Voltage": np.random.uniform(230, 250, len(rng)),
            "Global_intensity": np.random.uniform(0.5, 20.0, len(rng)),
            "Sub_metering_1": np.random.uniform(0, 50, len(rng)),
            "Sub_metering_2": np.random.uniform(0, 30, len(rng)),
            "Sub_metering_3": np.random.uniform(0, 20, len(rng)),
        },
        index=rng,
    )
    return df


class TestCleaner:
    def test_clean_preserves_valid_data(self, sample_minute_data):
        """Clean should not drop valid rows."""
        result = clean(sample_minute_data)
        assert len(result) == len(sample_minute_data)

    def test_clean_removes_negative_power(self, sample_minute_data):
        """Negative power values should be removed."""
        sample_minute_data.iloc[0, 0] = -1.0  # Global_active_power
        result = clean(sample_minute_data)
        assert len(result) < len(sample_minute_data)

    def test_clean_handles_missing_values(self, sample_minute_data):
        """Forward-fill should handle NaN values."""
        sample_minute_data.iloc[5:10, 0] = np.nan
        result = clean(sample_minute_data)
        assert not result["Global_active_power"].isna().any()

    def test_clean_removes_low_voltage(self, sample_minute_data):
        """Voltage ≤100 should be flagged as impossible."""
        sample_minute_data.iloc[0, 2] = 50.0  # Voltage
        result = clean(sample_minute_data)
        assert len(result) < len(sample_minute_data)


class TestResample:
    def test_resample_reduces_rows(self, sample_minute_data):
        """1-hour resampling of 1440 minutes → ~24 rows."""
        result = resample(sample_minute_data, freq="1h")
        assert len(result) == 24

    def test_resample_creates_unmetered(self, sample_minute_data):
        """Should compute Unmetered_consumption column."""
        result = resample(sample_minute_data, freq="1h")
        assert "Unmetered_consumption" in result.columns
        assert (result["Unmetered_consumption"] >= 0).all()

    def test_resample_custom_frequency(self, sample_minute_data):
        """Should support custom frequencies."""
        result = resample(sample_minute_data, freq="15min")
        assert len(result) == 96  # 1440 / 15


class TestFeatureEngineering:
    def test_add_time_features(self, sample_minute_data):
        """Should add cyclical time columns."""
        result = add_time_features(sample_minute_data)
        expected_cols = ["hour", "day_of_week", "month", "is_weekend",
                         "hour_sin", "hour_cos"]
        for col in expected_cols:
            assert col in result.columns

    def test_cyclical_encoding_bounds(self, sample_minute_data):
        """Sin/cos features should be in [-1, 1]."""
        result = add_time_features(sample_minute_data)
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_add_rolling_features(self, sample_minute_data):
        """Should create rolling mean and std columns."""
        result = add_rolling_features(sample_minute_data, windows=[6, 24])
        assert "Global_active_power_rolling_mean_6" in result.columns
        assert "Global_active_power_rolling_std_24" in result.columns

    def test_add_rate_of_change(self, sample_minute_data):
        """Should create diff and pct_change columns."""
        result = add_rate_of_change(sample_minute_data)
        assert "Global_active_power_diff" in result.columns
        assert "Global_active_power_pct_change" in result.columns
        # No infinities
        assert not np.isinf(result["Global_active_power_pct_change"]).any()

    def test_full_pipeline(self, sample_minute_data):
        """engineer_features should produce all expected columns."""
        result = engineer_features(sample_minute_data)
        assert len(result.columns) > len(sample_minute_data.columns) + 10
