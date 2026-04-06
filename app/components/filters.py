"""
Reusable filter widgets for the Streamlit UI.

Provides consistent date range, severity, and sub-meter filter
components that can be used across multiple pages.
"""

from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st


def date_range_filter(
    df: pd.DataFrame,
    key_prefix: str = "filter",
) -> tuple[str, str]:
    """
    Render a date range picker based on the data's time range.

    Returns (start_date, end_date) as ISO format strings.
    """
    if df.empty:
        today = date.today()
        return str(today - timedelta(days=30)), str(today)

    min_date = df.index.min().date()
    max_date = df.index.max().date()

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=90),
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_start",
        )
    with col2:
        end = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_end",
        )

    return str(start), str(end)


def severity_filter(key: str = "severity_filter") -> str:
    """Severity select box. Returns selected severity or 'All'."""
    return st.selectbox(
        "Severity Level",
        options=["All", "critical", "high", "medium", "low"],
        index=0,
        key=key,
    )


def sub_meter_selector(key: str = "submeter_selector") -> list[str]:
    """Multi-select for sub-meter columns."""
    options = [
        "Sub_metering_1 (Kitchen)",
        "Sub_metering_2 (Laundry)",
        "Sub_metering_3 (Water Heater & AC)",
        "Unmetered_consumption (Other)",
    ]
    selected = st.multiselect(
        "Sub-Meters",
        options=options,
        default=options,
        key=key,
    )
    # Map display names back to column names
    mapping = {
        "Sub_metering_1 (Kitchen)": "Sub_metering_1",
        "Sub_metering_2 (Laundry)": "Sub_metering_2",
        "Sub_metering_3 (Water Heater & AC)": "Sub_metering_3",
        "Unmetered_consumption (Other)": "Unmetered_consumption",
    }
    return [mapping[s] for s in selected if s in mapping]
