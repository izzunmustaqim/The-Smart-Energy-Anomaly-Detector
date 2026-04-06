"""
Data cleaning and resampling pipeline.

Handles missing values, type validation, and temporal resampling
from minute-level to the configured frequency (default 1 hour).
"""

import logging

import numpy as np
import pandas as pd

from processor.config import settings

logger = logging.getLogger(__name__)

# Expected columns from the UCI dataset after parsing
EXPECTED_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame.

    Steps:
    1. Validate expected columns exist
    2. Drop fully-duplicate rows
    3. Forward-fill missing values (bounded to 60-minute gaps)
    4. Drop any remaining NaN rows
    5. Remove physically impossible readings (negative power, zero voltage)

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with datetime index.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for resampling.
    """
    logger.info("Cleaning data — initial shape: %s", df.shape)

    # 1. Validate columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        logger.warning("Missing expected columns: %s", missing_cols)

    # 2. Drop duplicates
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_dupes = n_before - len(df)
    if n_dupes:
        logger.info("Removed %d duplicate timestamps", n_dupes)

    # 3. Forward-fill bounded to 60 minutes
    null_count = df.isnull().sum().sum()
    logger.info("Missing values before fill: %d", null_count)
    df = df.ffill(limit=60)

    # 4. Drop remaining NaNs
    n_before = len(df)
    df = df.dropna()
    logger.info("Dropped %d rows still containing NaN", n_before - len(df))

    # 5. Remove physically impossible readings
    if "Global_active_power" in df.columns:
        mask = (df["Global_active_power"] >= 0) & (df["Voltage"] > 100)
        removed = (~mask).sum()
        if removed:
            logger.info("Removed %d rows with impossible readings", removed)
        df = df[mask]

    logger.info("Cleaning complete — final shape: %s", df.shape)
    return df


def resample(
    df: pd.DataFrame,
    freq: str | None = None,
) -> pd.DataFrame:
    """
    Resample time-series to a coarser frequency.

    Power/intensity columns are averaged; sub-metering columns
    (energy in Wh) are summed.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with datetime index.
    freq : str or None
        Pandas offset alias (e.g. '1h', '15min', '1D').
        Defaults to ``settings.resample_freq``.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame.
    """
    freq = freq or settings.resample_freq
    logger.info("Resampling from minute-level to '%s'", freq)

    avg_cols = [c for c in ["Global_active_power", "Global_reactive_power",
                            "Voltage", "Global_intensity"] if c in df.columns]
    sum_cols = [c for c in ["Sub_metering_1", "Sub_metering_2",
                            "Sub_metering_3"] if c in df.columns]

    agg_map = {c: "mean" for c in avg_cols}
    agg_map.update({c: "sum" for c in sum_cols})

    df_resampled = df.resample(freq).agg(agg_map)
    df_resampled.dropna(inplace=True)

    # Compute "unmetered" consumption (everything not covered by sub-meters)
    if all(c in df_resampled.columns for c in ["Global_active_power",
                                                "Sub_metering_1",
                                                "Sub_metering_2",
                                                "Sub_metering_3"]):
        # global_active_power is kW (mean); sub_metering is Wh (sum)
        # Convert global kW-mean over 1h → Wh: kW * 1000
        minutes_in_freq = pd.Timedelta(freq).total_seconds() / 60
        df_resampled["Unmetered_consumption"] = (
            df_resampled["Global_active_power"] * 1000 / 60 * minutes_in_freq
            - df_resampled["Sub_metering_1"]
            - df_resampled["Sub_metering_2"]
            - df_resampled["Sub_metering_3"]
        ).clip(lower=0)

    logger.info("Resampled shape: %s", df_resampled.shape)
    return df_resampled
