"""
Isolation Forest anomaly detection model.

Uses scikit-learn's IsolationForest trained on multivariate
power-consumption features. Outputs normalized anomaly scores
in [0, 1] where 1 = most anomalous.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from processor.config import settings
from processor.models.base import AnomalyModel

logger = logging.getLogger(__name__)

# Feature columns used by the Isolation Forest
IF_FEATURE_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "Global_active_power_rolling_mean_24",
    "Global_active_power_rolling_std_24",
    "Global_active_power_diff",
    "Global_active_power_pct_change",
]


class IsolationForestModel(AnomalyModel):
    """Multivariate Isolation Forest for point-anomaly detection."""

    def __init__(self) -> None:
        self._model: IsolationForest | None = None
        self._feature_cols: list[str] = []

    @property
    def name(self) -> str:
        return "IsolationForest"

    def train(self, df: pd.DataFrame) -> None:
        """Fit the Isolation Forest on available features."""
        self._feature_cols = [c for c in IF_FEATURE_COLS if c in df.columns]
        if not self._feature_cols:
            raise ValueError("No matching feature columns found in DataFrame.")

        logger.info(
            "Training Isolation Forest on %d features × %d samples",
            len(self._feature_cols),
            len(df),
        )

        X = df[self._feature_cols].values

        self._model = IsolationForest(
            n_estimators=settings.if_n_estimators,
            contamination=settings.contamination,
            random_state=settings.if_random_state,
            n_jobs=-1,
        )
        self._model.fit(X)
        logger.info("Isolation Forest training complete.")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Score each row. Raw IF scores are in (-∞, 0] where more
        negative = more anomalous. We normalize to [0, 1].
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        X = df[self._feature_cols].values

        # decision_function: lower (more negative) = more anomalous
        raw_scores = self._model.decision_function(X)

        # Normalize to [0, 1]:  most anomalous → 1
        scores_min = raw_scores.min()
        scores_max = raw_scores.max()
        denom = scores_max - scores_min
        if denom == 0:
            normalized = np.zeros_like(raw_scores)
        else:
            normalized = 1 - (raw_scores - scores_min) / denom

        return pd.Series(normalized, index=df.index, name="if_anomaly_score")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self._model,
            "feature_cols": self._feature_cols,
        }
        joblib.dump(artifact, path)
        logger.info("Isolation Forest saved to %s", path)

    def load(self, path: Path) -> None:
        artifact = joblib.load(path)
        self._model = artifact["model"]
        self._feature_cols = artifact["feature_cols"]
        logger.info("Isolation Forest loaded from %s", path)
