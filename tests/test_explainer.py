"""Tests for the explainability engine."""

import numpy as np
import pandas as pd
import pytest

from processor.explainability.explainer import explain_anomalies


@pytest.fixture
def sample_data_and_scores():
    """Create sample data with scores for explanation testing."""
    # 4 weeks ensures each (DOW, hour) slot has ~4 baseline points
    rng = pd.date_range("2024-01-01", periods=672, freq="1h")
    np.random.seed(42)

    # Use a tight range (1.0–2.0) so a 12.0 spike is unambiguous
    df = pd.DataFrame(
        {
            "Global_active_power": np.random.uniform(1.0, 2.0, len(rng)),
            "Sub_metering_1": np.random.uniform(0, 30, len(rng)),
            "Sub_metering_2": np.random.uniform(0, 20, len(rng)),
            "Sub_metering_3": np.random.uniform(0, 15, len(rng)),
            "Unmetered_consumption": np.random.uniform(0, 40, len(rng)),
            "hour": rng.hour,
            "day_of_week": rng.dayofweek,
        },
        index=rng,
    )

    # Inject an obvious anomaly — well above the 1.0–2.0 range
    df.iloc[50, 0] = 12.0  # Huge spike on Global_active_power

    scores_df = pd.DataFrame(
        {
            "IsolationForest": np.random.uniform(0.0, 0.5, len(rng)),
            "ensemble_score": np.random.uniform(0.0, 0.5, len(rng)),
            "is_anomaly": 0,
        },
        index=rng,
    )
    scores_df.iloc[50, 1] = 0.95  # Mark it as highly anomalous
    scores_df.iloc[50, 2] = 1

    return df, scores_df


class TestExplainer:
    def test_explain_produces_explanations(self, sample_data_and_scores):
        df, scores_df = sample_data_and_scores
        anomaly_indices = scores_df[scores_df["is_anomaly"] == 1].index

        explanations = explain_anomalies(df, scores_df, anomaly_indices)

        assert len(explanations) == len(anomaly_indices)
        assert len(explanations) > 0

    def test_explanation_structure(self, sample_data_and_scores):
        df, scores_df = sample_data_and_scores
        anomaly_indices = scores_df[scores_df["is_anomaly"] == 1].index

        explanations = explain_anomalies(df, scores_df, anomaly_indices)
        exp = explanations[0]

        required_keys = [
            "timestamp", "severity", "severity_emoji", "ensemble_score",
            "actual_value", "expected_mean", "expected_min", "expected_max",
            "deviation_pct", "direction", "contributing_factors",
            "human_readable_text",
        ]
        for key in required_keys:
            assert key in exp, f"Missing key: {key}"

    def test_spike_detected_as_spike_direction(self, sample_data_and_scores):
        df, scores_df = sample_data_and_scores
        anomaly_indices = scores_df[scores_df["is_anomaly"] == 1].index

        explanations = explain_anomalies(df, scores_df, anomaly_indices)
        exp = explanations[0]

        assert exp["direction"] == "spike"
        assert exp["actual_value"] == 12.0
        assert exp["deviation_pct"] > 0

    def test_severity_labels(self, sample_data_and_scores):
        df, scores_df = sample_data_and_scores
        anomaly_indices = scores_df[scores_df["is_anomaly"] == 1].index

        explanations = explain_anomalies(df, scores_df, anomaly_indices)

        for exp in explanations:
            assert exp["severity"] in ["critical", "high", "medium", "low"]
            assert exp["severity_emoji"] in ["🔴", "🟠", "🟡", "🟢"]

    def test_human_readable_text_not_empty(self, sample_data_and_scores):
        df, scores_df = sample_data_and_scores
        anomaly_indices = scores_df[scores_df["is_anomaly"] == 1].index

        explanations = explain_anomalies(df, scores_df, anomaly_indices)

        for exp in explanations:
            assert len(exp["human_readable_text"]) > 20
