"""
Smart Alerts page — AI-explained anomaly alerts.

The hero feature of the application. Displays a filterable list
of anomaly cards with contextual explanations showing *why* each
data point was flagged as abnormal.
"""

import pandas as pd
import streamlit as st

from app.components.alert_card import render_alert_card
from app.components.filters import date_range_filter, severity_filter
from app.services.data_service import get_anomalies, load_consumption_data


def render() -> None:
    """Render the Smart Alerts page."""
    st.markdown(
        """
        <h1 style="
            background: linear-gradient(135deg, #EF4444, #F59E0B);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        ">🧠 Smart Alerts</h1>
        <p style="color: #94A3B8; margin-bottom: 1.5rem;">
            AI-powered anomaly detection with contextual explanations.
            Each alert explains <em>why</em> a specific time period was flagged as unusual.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Filters ────────────────────────────────────────────────────
    with st.container():
        st.markdown(
            """<div style="
                background: rgba(30,41,59,0.6);
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 20px;
                border: 1px solid rgba(148,163,184,0.08);
            ">""",
            unsafe_allow_html=True,
        )
        st.markdown("##### 🔍 Filters")
        filter_col1, filter_col2 = st.columns([3, 1])

        with filter_col1:
            df_consumption = load_consumption_data()
            start_date, end_date = date_range_filter(
                df_consumption, key_prefix="alerts"
            )

        with filter_col2:
            severity = severity_filter(key="alerts_severity")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Load filtered anomalies ────────────────────────────────────
    anomalies = get_anomalies(
        severity=severity if severity != "All" else None,
        start_date=start_date,
        end_date=end_date,
    )

    # ── Summary bar ────────────────────────────────────────────────
    if not anomalies.empty:
        total = len(anomalies)
        critical = (anomalies["severity"] == "critical").sum()
        high = (anomalies["severity"] == "high").sum()
        medium = (anomalies["severity"] == "medium").sum()
        low = (anomalies["severity"] == "low").sum()

        st.markdown(
            f"""
            <div style="
                display: flex; gap: 16px; margin-bottom: 24px;
                flex-wrap: wrap;
            ">
                <div style="background: rgba(239,68,68,0.1); color: #EF4444;
                            padding: 8px 16px; border-radius: 8px; font-weight: 600;">
                    🔴 {critical} Critical
                </div>
                <div style="background: rgba(249,115,22,0.1); color: #F97316;
                            padding: 8px 16px; border-radius: 8px; font-weight: 600;">
                    🟠 {high} High
                </div>
                <div style="background: rgba(245,158,11,0.1); color: #F59E0B;
                            padding: 8px 16px; border-radius: 8px; font-weight: 600;">
                    🟡 {medium} Medium
                </div>
                <div style="background: rgba(16,185,129,0.1); color: #10B981;
                            padding: 8px 16px; border-radius: 8px; font-weight: 600;">
                    🟢 {low} Low
                </div>
                <div style="background: rgba(99,102,241,0.1); color: #6366F1;
                            padding: 8px 16px; border-radius: 8px; font-weight: 600;">
                    📊 {total} Total
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Render alert cards ─────────────────────────────────────
        # Paginate for performance
        page_size = 10
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            key="alerts_page",
        )

        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total)
        st.caption(f"Showing alerts {start_idx + 1}–{end_idx} of {total}")

        for i in range(start_idx, end_idx):
            row = anomalies.iloc[i]
            render_alert_card(
                alert=row,
                consumption_df=df_consumption,
                index=i,
            )

    else:
        st.info(
            "🎉 No anomalies found for the selected filters. "
            "Try expanding the date range or changing the severity filter."
        )
