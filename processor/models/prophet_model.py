"""
Prophet-based time-series forecasting model.

Trains a Prophet model on Global_active_power to learn the
"normal" energy rhythm including daily, weekly, and yearly
seasonality. Anomalies are data points whose actual values
deviate significantly from the forecast.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from processor.config import settings
from processor.models.base import AnomalyModel

logger = logging.getLogger(__name__)


class ProphetModel(AnomalyModel):
    """Prophet-based trend/seasonality model for anomaly detection."""

    def __init__(self) -> None:
        self._model = None
        self._forecast: pd.DataFrame | None = None
        self._residual_std: float = 1.0

    @property
    def name(self) -> str:
        return "Prophet"

    def train(self, df: pd.DataFrame) -> None:
        """
        Fit Prophet on the Global_active_power time-series.

        Prophet requires columns ['ds', 'y'] so we reshape the
        datetime index accordingly.
        """
        # Lazy import — Prophet is heavy and slow to load
        from prophet import Prophet

        logger.info("Training Prophet on %d data points …", len(df))

        # Prepare Prophet input format
        prophet_df = pd.DataFrame({
            "ds": df.index,
            "y": df["Global_active_power"].values,
        })

        self._model = Prophet(
            uncertainty_samples=settings.prophet_uncertainty_samples,
            changepoint_prior_scale=settings.prophet_changepoint_prior_scale,
            seasonality_mode=settings.prophet_seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )

        # Suppress verbose cmdstan output
        self._model.fit(prophet_df, iter=300)
        logger.info("Prophet training complete.")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate anomaly scores based on forecast residuals.

        Score = |actual - yhat| / residual_std, normalized to [0, 1].
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        # Create future dataframe matching input timestamps
        future_df = pd.DataFrame({"ds": df.index})
        forecast = self._model.predict(future_df)

        self._forecast = forecast.set_index("ds")

        # Compute residuals
        actual = df["Global_active_power"].values
        predicted = forecast["yhat"].values
        residuals = np.abs(actual - predicted)

        # Use MAD (median absolute deviation) for robust scaling
        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual))
        self._residual_std = mad * 1.4826  # scale to match std dev

        if self._residual_std == 0:
            self._residual_std = 1.0

        # Normalize: higher residual → higher anomaly score
        z_scores = residuals / self._residual_std
        # Clip and scale to [0, 1]
        max_z = max(z_scores.max(), 1.0)
        normalized = np.clip(z_scores / max_z, 0, 1)

        return pd.Series(normalized, index=df.index, name="prophet_anomaly_score")

    def get_forecast(self) -> pd.DataFrame | None:
        """Return the last forecast DataFrame (for visualization)."""
        return self._forecast

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self._model,
            "residual_std": self._residual_std,
        }
        joblib.dump(artifact, path)
        logger.info("Prophet model saved to %s", path)

    def load(self, path: Path) -> None:
        artifact = joblib.load(path)
        self._model = artifact["model"]
        self._residual_std = artifact["residual_std"]
        logger.info("Prophet model loaded from %s", path)
