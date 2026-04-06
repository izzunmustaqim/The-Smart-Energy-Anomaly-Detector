"""
Processor configuration using pydantic-settings.

Loads settings from environment variables and .env file with
sensible defaults for all parameters.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessorSettings(BaseSettings):
    """Configuration for the data processing pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Paths ──────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    db_path: Path = Path("db/anomalies.db")

    # ── Data Processing ────────────────────────────────────────────
    resample_freq: str = "1h"
    """Resampling frequency for raw minute-level data (pandas offset alias)."""

    # ── Isolation Forest ───────────────────────────────────────────
    contamination: float = 0.02
    """Expected fraction of anomalies in the dataset (0.0 – 0.5)."""

    if_n_estimators: int = 200
    """Number of trees in the Isolation Forest ensemble."""

    if_random_state: int = 42
    """Random seed for reproducibility."""

    # ── Prophet ────────────────────────────────────────────────────
    prophet_uncertainty_samples: int = 1000
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_mode: str = "multiplicative"

    # ── Ensemble ───────────────────────────────────────────────────
    if_weight: float = 0.5
    """Weight assigned to Isolation Forest scores in the ensemble."""

    prophet_weight: float = 0.5
    """Weight assigned to Prophet residual scores in the ensemble."""

    anomaly_threshold_percentile: float = 95.0
    """Percentile cutoff for labelling a data point as anomalous."""

    # ── Derived helpers ────────────────────────────────────────────
    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


# Module-level singleton
settings = ProcessorSettings()
