"""
SQLite database manager for anomaly persistence.

Manages two tables:
- `anomalies`: detected anomalies with explanations
- `model_runs`: metadata about each training/detection run

Uses the stdlib sqlite3 module — no ORM needed for this scope.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from processor.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """CRUD operations for the anomaly detection database."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Schema ─────────────────────────────────────────────────────
    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    severity_emoji TEXT,
                    ensemble_score REAL NOT NULL,
                    actual_value REAL,
                    expected_mean REAL,
                    expected_min REAL,
                    expected_max REAL,
                    deviation_pct REAL,
                    direction TEXT,
                    contributing_factors TEXT,
                    human_readable_text TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    trained_at TEXT NOT NULL,
                    parameters TEXT,
                    metrics TEXT,
                    artifact_path TEXT,
                    n_anomalies INTEGER,
                    threshold REAL
                );

                CREATE INDEX IF NOT EXISTS idx_anomalies_ts
                    ON anomalies(timestamp);
                CREATE INDEX IF NOT EXISTS idx_anomalies_severity
                    ON anomalies(severity);
            """)
        logger.info("Database schema initialized at %s", self._db_path)

    # ── Connection helper ──────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── Anomaly CRUD ───────────────────────────────────────────────
    def insert_anomalies(self, explanations: list[dict]) -> int:
        """
        Bulk-insert anomaly explanations.

        Returns the number of rows inserted.
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                str(e["timestamp"]),
                e["severity"],
                e["severity_emoji"],
                e["ensemble_score"],
                e["actual_value"],
                e["expected_mean"],
                e["expected_min"],
                e["expected_max"],
                e["deviation_pct"],
                e["direction"],
                json.dumps(e["contributing_factors"]),
                e["human_readable_text"],
                now,
            )
            for e in explanations
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO anomalies (
                    timestamp, severity, severity_emoji, ensemble_score,
                    actual_value, expected_mean, expected_min, expected_max,
                    deviation_pct, direction, contributing_factors,
                    human_readable_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

        logger.info("Inserted %d anomaly records.", len(rows))
        return len(rows)

    def get_anomalies(
        self,
        severity: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Query anomalies with optional filters.

        Returns a DataFrame sorted by timestamp descending.
        """
        query = "SELECT * FROM anomalies WHERE 1=1"
        params: list = []

        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY ensemble_score DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if "contributing_factors" in df.columns:
            df["contributing_factors"] = df["contributing_factors"].apply(
                lambda x: json.loads(x) if x else []
            )
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def clear_anomalies(self) -> None:
        """Delete all anomaly records (for re-processing)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM anomalies")
        logger.info("Cleared all anomaly records.")

    # ── Model Run CRUD ─────────────────────────────────────────────
    def insert_model_run(
        self,
        model_name: str,
        parameters: dict | None = None,
        metrics: dict | None = None,
        artifact_path: str | None = None,
        n_anomalies: int = 0,
        threshold: float = 0.0,
    ) -> int:
        """Record metadata for a model training/detection run."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO model_runs (
                    model_name, trained_at, parameters, metrics,
                    artifact_path, n_anomalies, threshold
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    now,
                    json.dumps(parameters) if parameters else None,
                    json.dumps(metrics) if metrics else None,
                    artifact_path,
                    n_anomalies,
                    threshold,
                ),
            )
            run_id = cursor.lastrowid

        logger.info("Recorded model run: %s (id=%d)", model_name, run_id)
        return run_id

    def get_model_runs(self) -> pd.DataFrame:
        """Return all model run records."""
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM model_runs ORDER BY trained_at DESC", conn
            )
