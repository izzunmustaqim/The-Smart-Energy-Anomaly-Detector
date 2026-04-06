"""
Data service layer for the Streamlit app.

Reads processed Parquet files and queries the SQLite database.
Uses Streamlit's caching for performance.
"""

import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import app_settings

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)
def load_consumption_data() -> pd.DataFrame:
    """Load the hourly consumption Parquet file."""
    path = app_settings.processed_data_dir / "consumption_hourly.parquet"
    if not path.exists():
        st.error(f"Processed data not found at {path}. Run the processor first.")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=300)
def load_anomaly_scores() -> pd.DataFrame:
    """Load the anomaly scores Parquet file."""
    path = app_settings.processed_data_dir / "anomaly_scores.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=60)
def get_anomalies(
    severity: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Query anomalies from the SQLite database."""
    db_path = app_settings.db_path
    if not db_path.exists():
        return pd.DataFrame()

    query = "SELECT * FROM anomalies WHERE 1=1"
    params: list = []

    if severity and severity != "All":
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

    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if "contributing_factors" in df.columns:
        df["contributing_factors"] = df["contributing_factors"].apply(
            lambda x: json.loads(x) if x else []
        )
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


@st.cache_data(ttl=300)
def get_summary_stats() -> dict:
    """Compute summary statistics for the dashboard KPIs."""
    df = load_consumption_data()
    anomalies = get_anomalies()

    if df.empty:
        return {
            "total_records": 0,
            "date_range_start": None,
            "date_range_end": None,
            "avg_power_kw": 0,
            "peak_power_kw": 0,
            "total_anomalies": 0,
            "critical_anomalies": 0,
        }

    return {
        "total_records": len(df),
        "date_range_start": df.index.min(),
        "date_range_end": df.index.max(),
        "avg_power_kw": float(df["Global_active_power"].mean()),
        "peak_power_kw": float(df["Global_active_power"].max()),
        "total_anomalies": len(anomalies),
        "critical_anomalies": int(
            (anomalies["severity"] == "critical").sum() if not anomalies.empty else 0
        ),
    }
