"""
Temporal feature engineering for anomaly detection.

Extracts cyclical time features, rolling statistics, and
rate-of-change metrics to give the ML models rich context
about the energy consumption patterns.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features derived from the datetime index.

    Features created:
    - hour, day_of_week, month, day_of_year
    - is_weekend (binary)
    - hour_sin, hour_cos (cyclical encoding)
    - dow_sin, dow_cos (cyclical encoding)
    - month_sin, month_cos (cyclical encoding)
    """
    logger.info("Adding time features …")

    df = df.copy()
    idx = df.index

    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["day_of_year"] = idx.dayofyear
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Cyclical encoding — prevents the model from seeing midnight (0)
    # and 11 PM (23) as maximally distant
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling statistics for the target column.

    Parameters
    ----------
    target_col : str
        Column to compute rolling stats on.
    windows : list[int]
        Window sizes in number of periods (hours if hourly data).
        Default: [6, 24, 168] → 6h, 24h, 7-day.
    """
    if windows is None:
        windows = [6, 24, 168]

    if target_col not in df.columns:
        logger.warning("Target column '%s' not found — skipping rolling features", target_col)
        return df

    df = df.copy()
    logger.info("Adding rolling features for '%s' with windows %s", target_col, windows)

    for w in windows:
        roll = df[target_col].rolling(window=w, min_periods=1)
        df[f"{target_col}_rolling_mean_{w}"] = roll.mean()
        df[f"{target_col}_rolling_std_{w}"] = roll.std().fillna(0)

    return df


def add_rate_of_change(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
) -> pd.DataFrame:
    """
    Add rate-of-change (first difference) and percentage change.
    """
    if target_col not in df.columns:
        return df

    df = df.copy()
    logger.info("Adding rate-of-change features for '%s'", target_col)

    df[f"{target_col}_diff"] = df[target_col].diff().fillna(0)
    df[f"{target_col}_pct_change"] = (
        df[target_col].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    )

    return df


def engineer_features(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Applies all feature transformations in sequence.
    """
    df = add_time_features(df)
    df = add_rolling_features(df, target_col=target_col)
    df = add_rate_of_change(df, target_col=target_col)

    logger.info("Feature engineering complete — %d columns total", len(df.columns))
    return df
