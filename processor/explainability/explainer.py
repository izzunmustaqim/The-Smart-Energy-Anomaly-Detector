"""
Contextual explanation generator for detected anomalies.

For each flagged anomaly, produces a human-readable explanation
that describes *why* the reading is unusual by comparing it
against the historical baseline for that specific time slot.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Day-of-week names for readable output
DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


def _severity_label(score: float) -> str:
    """Map an ensemble score to a severity label."""
    if score >= 0.90:
        return "critical"
    elif score >= 0.75:
        return "high"
    elif score >= 0.50:
        return "medium"
    return "low"


def _severity_emoji(label: str) -> str:
    return {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
    }.get(label, "⚪")


def explain_anomalies(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    anomaly_indices: pd.DatetimeIndex,
) -> list[dict]:
    """
    Generate contextual explanations for each anomaly.

    Parameters
    ----------
    df : pd.DataFrame
        The feature-engineered consumption data.
    scores_df : pd.DataFrame
        Model scores DataFrame (from DetectionResult.scores).
    anomaly_indices : pd.DatetimeIndex
        Timestamps of detected anomalies.

    Returns
    -------
    list[dict]
        One explanation dict per anomaly with keys:
        - timestamp, severity, severity_emoji, actual_value,
          expected_mean, expected_min, expected_max, deviation_pct,
          direction, contributing_factors, human_readable_text
    """
    logger.info("Generating explanations for %d anomalies …", len(anomaly_indices))

    explanations: list[dict] = []

    for ts in anomaly_indices:
        row = df.loc[ts]
        score_row = scores_df.loc[ts]

        actual = row.get("Global_active_power", np.nan)
        hour = ts.hour
        dow = ts.dayofweek
        day_name = DAY_NAMES[dow]

        # ── Compute historical baseline for this (day_of_week, hour) slot
        mask = (df.index.dayofweek == dow) & (df.index.hour == hour)
        baseline_values = df.loc[mask, "Global_active_power"]

        if len(baseline_values) > 0:
            expected_mean = baseline_values.mean()
            expected_std = baseline_values.std()
            expected_min = baseline_values.quantile(0.10)
            expected_max = baseline_values.quantile(0.90)
        else:
            expected_mean = actual
            expected_std = 0
            expected_min = actual
            expected_max = actual

        # ── Deviation analysis
        deviation = actual - expected_mean
        deviation_pct = (
            (deviation / expected_mean * 100) if expected_mean != 0 else 0
        )
        direction = "spike" if deviation > 0 else "drop"

        # ── Sub-meter contribution analysis
        contributing_factors = []
        for meter, label in [
            ("Sub_metering_1", "Kitchen (dishwasher, oven, microwave)"),
            ("Sub_metering_2", "Laundry (washer, dryer, fridge)"),
            ("Sub_metering_3", "Water heater & AC"),
        ]:
            if meter in df.columns:
                meter_val = row.get(meter, 0)
                meter_baseline = df.loc[mask, meter].mean() if mask.sum() > 0 else 0
                if meter_baseline > 0 and meter_val > meter_baseline * 1.5:
                    factor_deviation = (meter_val - meter_baseline) / meter_baseline * 100
                    contributing_factors.append(
                        f"{label}: {meter_val:.0f} Wh "
                        f"(+{factor_deviation:.0f}% above normal)"
                    )

        if "Unmetered_consumption" in df.columns:
            unmetered = row.get("Unmetered_consumption", 0)
            unmetered_baseline = (
                df.loc[mask, "Unmetered_consumption"].mean() if mask.sum() > 0 else 0
            )
            if unmetered_baseline > 0 and unmetered > unmetered_baseline * 1.5:
                factor_dev = (unmetered - unmetered_baseline) / unmetered_baseline * 100
                contributing_factors.append(
                    f"Unmetered equipment: {unmetered:.0f} Wh "
                    f"(+{factor_dev:.0f}% above normal)"
                )

        # ── Severity
        ensemble_score = float(score_row.get("ensemble_score", 0))
        severity = _severity_label(ensemble_score)
        emoji = _severity_emoji(severity)

        # ── Build human-readable explanation
        text_parts = [
            f"{emoji} {severity.upper()} anomaly detected on "
            f"{day_name} at {hour:02d}:00.",
            f"Power consumption was {actual:.2f} kW — "
            f"the typical {day_name} {hour:02d}:00 range is "
            f"{expected_min:.2f}–{expected_max:.2f} kW "
            f"(mean {expected_mean:.2f} kW).",
        ]

        if abs(deviation_pct) > 5:
            text_parts.append(
                f"This is a {abs(deviation_pct):.0f}% {direction} "
                f"from the expected value."
            )

        if contributing_factors:
            text_parts.append("Contributing factors:")
            for factor in contributing_factors:
                text_parts.append(f"  • {factor}")
        else:
            if direction == "spike":
                text_parts.append(
                    "No single sub-meter dominates — this may indicate "
                    "multiple appliances running simultaneously or an "
                    "unusual load on unmetered circuits."
                )
            else:
                text_parts.append(
                    "Power dropped below expected levels — possible "
                    "equipment shutdown, outage, or occupancy change."
                )

        explanation = {
            "timestamp": ts,
            "severity": severity,
            "severity_emoji": emoji,
            "ensemble_score": ensemble_score,
            "actual_value": float(actual),
            "expected_mean": float(expected_mean),
            "expected_min": float(expected_min),
            "expected_max": float(expected_max),
            "deviation_pct": float(deviation_pct),
            "direction": direction,
            "contributing_factors": contributing_factors,
            "human_readable_text": "\n".join(text_parts),
        }
        explanations.append(explanation)

    logger.info("Generated %d explanations.", len(explanations))
    return explanations
