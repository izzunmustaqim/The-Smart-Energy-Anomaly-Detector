"""
Main pipeline orchestrator for the data processor.

Coordinates the full processing pipeline:
1. Ingest raw data (UCI dataset)
2. Clean and resample
3. Engineer features
4. Train anomaly detection models
5. Run detection and generate explanations
6. Persist results to SQLite + Parquet

This module is the entry point for the processor Docker container.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from processor.config import settings
from processor.detection.detector import AnomalyDetector
from processor.explainability.explainer import explain_anomalies
from processor.ingestion.uci_adapter import UCIAdapter
from processor.models.isolation_forest import IsolationForestModel
from processor.models.prophet_model import ProphetModel
from processor.preprocessing.cleaner import clean, resample
from processor.preprocessing.features import engineer_features
from processor.storage.db_manager import DatabaseManager

# ── Logging setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Execute the full data processing and anomaly detection pipeline."""
    logger.info("=" * 60)
    logger.info("SMART ENERGY ANOMALY DETECTOR — Processing Pipeline")
    logger.info("=" * 60)

    # ── Ensure directories exist ───────────────────────────────────
    for d in [settings.raw_data_dir, settings.processed_data_dir, settings.models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Ingest ─────────────────────────────────────────────
    logger.info("STEP 1/6: Data Ingestion")
    adapter = UCIAdapter()
    df_raw = adapter.load()

    # ── Step 2: Clean ──────────────────────────────────────────────
    logger.info("STEP 2/6: Data Cleaning")
    df_clean = clean(df_raw)

    # ── Step 3: Resample ───────────────────────────────────────────
    logger.info("STEP 3/6: Resampling to '%s'", settings.resample_freq)
    df_resampled = resample(df_clean)

    # Save processed data as Parquet for the Streamlit app
    processed_path = settings.processed_data_dir / "consumption_hourly.parquet"
    df_resampled.to_parquet(processed_path, engine="pyarrow")
    logger.info("Saved processed data → %s", processed_path)

    # ── Step 4: Feature Engineering ────────────────────────────────
    logger.info("STEP 4/6: Feature Engineering")
    df_features = engineer_features(df_resampled)

    # ── Step 5: Train & Detect ─────────────────────────────────────
    logger.info("STEP 5/6: Model Training & Anomaly Detection")
    if_model = IsolationForestModel()
    prophet_model = ProphetModel()

    detector = AnomalyDetector(models=[if_model, prophet_model])
    detector.train_all(df_features)

    result = detector.detect(df_features)

    # Save model artifacts
    if_model.save(settings.models_dir / "isolation_forest.joblib")
    prophet_model.save(settings.models_dir / "prophet.joblib")

    # Save scores as Parquet (for Exploration tab overlays)
    scores_path = settings.processed_data_dir / "anomaly_scores.parquet"
    result.scores.to_parquet(scores_path, engine="pyarrow")
    logger.info("Saved anomaly scores → %s", scores_path)

    # ── Step 6: Explain & Persist ──────────────────────────────────
    logger.info("STEP 6/6: Generating Explanations & Persisting Results")
    explanations = explain_anomalies(
        df=df_features,
        scores_df=result.scores,
        anomaly_indices=result.anomalies.index,
    )

    db = DatabaseManager()
    db.clear_anomalies()  # Idempotent: clear previous run results
    db.insert_anomalies(explanations)

    # Record model run metadata
    for model in [if_model, prophet_model]:
        db.insert_model_run(
            model_name=model.name,
            n_anomalies=len(result.anomalies),
            threshold=result.threshold,
        )

    # ── Summary ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Total data points:  %d", len(df_features))
    logger.info("  Anomalies detected: %d", len(result.anomalies))
    logger.info("  Threshold:          %.4f", result.threshold)
    logger.info("  Results saved to:   %s", settings.db_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
