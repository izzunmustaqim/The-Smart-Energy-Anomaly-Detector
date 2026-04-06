"""
Anomaly detection orchestrator.

Runs multiple AnomalyModel implementations, normalizes their
scores, computes a weighted ensemble, and classifies data points
as anomalous or normal using a percentile-based threshold.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from processor.config import settings
from processor.models.base import AnomalyModel

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Container for the full detection output."""

    scores: pd.DataFrame
    """Per-model scores + ensemble score for every timestamp."""

    anomalies: pd.DataFrame
    """Subset of rows flagged as anomalous, sorted by severity."""

    threshold: float
    """The score cutoff used for classification."""


class AnomalyDetector:
    """
    Ensemble anomaly detector.

    Runs each registered model, normalizes outputs to [0, 1],
    and combines them with configurable weights.
    """

    def __init__(
        self,
        models: list[AnomalyModel],
        weights: dict[str, float] | None = None,
    ) -> None:
        self._models = models
        self._weights = weights or {
            "IsolationForest": settings.if_weight,
            "Prophet": settings.prophet_weight,
        }

    def train_all(self, df: pd.DataFrame) -> None:
        """Train every registered model."""
        for model in self._models:
            logger.info("Training model: %s", model.name)
            model.train(df)

    def detect(self, df: pd.DataFrame) -> DetectionResult:
        """
        Run all models, combine scores, and flag anomalies.

        Returns
        -------
        DetectionResult
            Contains per-model scores, anomaly subset, and threshold.
        """
        score_frames: dict[str, pd.Series] = {}

        for model in self._models:
            logger.info("Running predictions with %s", model.name)
            scores = model.predict(df)
            score_frames[model.name] = scores

        # Build scores DataFrame
        scores_df = pd.DataFrame(score_frames, index=df.index)

        # Weighted ensemble
        ensemble = np.zeros(len(df))
        total_weight = 0
        for model_name, series in score_frames.items():
            w = self._weights.get(model_name, 1.0)
            ensemble += series.values * w
            total_weight += w

        if total_weight > 0:
            ensemble /= total_weight

        scores_df["ensemble_score"] = ensemble

        # Determine threshold via percentile
        threshold = float(
            np.percentile(ensemble, settings.anomaly_threshold_percentile)
        )
        scores_df["is_anomaly"] = (ensemble >= threshold).astype(int)

        # Extract and sort anomalies
        anomalies = scores_df[scores_df["is_anomaly"] == 1].copy()
        anomalies.sort_values("ensemble_score", ascending=False, inplace=True)

        logger.info(
            "Detection complete: %d anomalies out of %d points (threshold=%.4f)",
            len(anomalies),
            len(df),
            threshold,
        )

        return DetectionResult(
            scores=scores_df,
            anomalies=anomalies,
            threshold=threshold,
        )
