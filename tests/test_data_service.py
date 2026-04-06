"""Tests for the SQLite database manager."""

import json

import pandas as pd
import pytest

from processor.storage.db_manager import DatabaseManager


@pytest.fixture
def db(tmp_path):
    """Create a temporary database."""
    return DatabaseManager(db_path=tmp_path / "test.db")


@pytest.fixture
def sample_explanations():
    """Sample anomaly explanation records."""
    return [
        {
            "timestamp": pd.Timestamp("2024-01-15 14:00:00"),
            "severity": "high",
            "severity_emoji": "🟠",
            "ensemble_score": 0.85,
            "actual_value": 8.4,
            "expected_mean": 2.9,
            "expected_min": 2.1,
            "expected_max": 3.8,
            "deviation_pct": 189.7,
            "direction": "spike",
            "contributing_factors": ["Water heater: +140% above normal"],
            "human_readable_text": "High anomaly on Tuesday at 14:00.",
        },
        {
            "timestamp": pd.Timestamp("2024-02-20 03:00:00"),
            "severity": "low",
            "severity_emoji": "🟢",
            "ensemble_score": 0.45,
            "actual_value": 0.1,
            "expected_mean": 0.8,
            "expected_min": 0.3,
            "expected_max": 1.2,
            "deviation_pct": -87.5,
            "direction": "drop",
            "contributing_factors": [],
            "human_readable_text": "Low anomaly on Tuesday at 03:00.",
        },
    ]


class TestDatabaseManager:
    def test_init_creates_tables(self, db, tmp_path):
        """Database init should create the schema."""
        assert (tmp_path / "test.db").exists()

    def test_insert_and_query_anomalies(self, db, sample_explanations):
        """Should insert and retrieve anomaly records."""
        count = db.insert_anomalies(sample_explanations)
        assert count == 2

        result = db.get_anomalies()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_filter_by_severity(self, db, sample_explanations):
        db.insert_anomalies(sample_explanations)

        highs = db.get_anomalies(severity="high")
        assert len(highs) == 1
        assert highs.iloc[0]["severity"] == "high"

    def test_clear_anomalies(self, db, sample_explanations):
        db.insert_anomalies(sample_explanations)
        db.clear_anomalies()

        result = db.get_anomalies()
        assert len(result) == 0

    def test_insert_model_run(self, db):
        run_id = db.insert_model_run(
            model_name="IsolationForest",
            parameters={"n_estimators": 200},
            n_anomalies=42,
            threshold=0.85,
        )
        assert run_id > 0

        runs = db.get_model_runs()
        assert len(runs) == 1
        assert runs.iloc[0]["model_name"] == "IsolationForest"

    def test_contributing_factors_json_roundtrip(self, db, sample_explanations):
        """Contributing factors should survive JSON serialization."""
        db.insert_anomalies(sample_explanations)
        result = db.get_anomalies()

        factors = result.iloc[0]["contributing_factors"]
        assert isinstance(factors, list)
        assert len(factors) == 1
        assert "Water heater" in factors[0]
