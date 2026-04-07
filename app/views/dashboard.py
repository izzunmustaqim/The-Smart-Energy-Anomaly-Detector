"""
Dashboard page — Overview with KPIs, trends, and summaries.
"""

import pandas as pd
import streamlit as st

from app.components.charts import (
    consumption_trend,
    hourly_heatmap,
    severity_distribution,
    sub_metering_donut,
)
from app.services.data_service import (
    get_anomalies,
    get_summary_stats,
    load_anomaly_scores,
    load_consumption_data,
)


def render() -> None:
    """Render the main dashboard page."""
    st.markdown(
        """
        <h1 style="
            background: linear-gradient(135deg, #6366F1, #06B6D4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        ">📊 Energy Dashboard</h1>
        <p style="color: #94A3B8; margin-bottom: 2rem;">
            Real-time overview of household energy consumption and detected anomalies.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI Cards ──────────────────────────────────────────────────
    stats = get_summary_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            _kpi_card(
                "⚡ Avg Power",
                f"{stats['avg_power_kw']:.2f} kW",
                "Hourly average",
                "#6366F1",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            _kpi_card(
                "🔺 Peak Power",
                f"{stats['peak_power_kw']:.2f} kW",
                "Maximum recorded",
                "#F59E0B",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            _kpi_card(
                "🚨 Anomalies",
                f"{stats['total_anomalies']}",
                f"{stats['critical_anomalies']} critical",
                "#EF4444",
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            _kpi_card(
                "📅 Data Points",
                f"{stats['total_records']:,}",
                "Hourly readings",
                "#10B981",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Consumption Trend ──────────────────────────────────────────
    df = load_consumption_data()
    scores = load_anomaly_scores()

    if not df.empty:
        # Show last 30 days by default for performance
        last_30d = df.loc[df.index >= df.index.max() - pd.Timedelta("30D")]
        scores_30d = scores.loc[scores.index.isin(last_30d.index)].get(
            "is_anomaly", pd.Series(dtype="float64")
        ) if not scores.empty else None

        st.plotly_chart(
            consumption_trend(
                last_30d,
                title="Energy Consumption — Last 30 Days",
                anomaly_scores=scores_30d,
            ),
            use_container_width=True,
        )

        # ── Two-column layout ─────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.plotly_chart(
                hourly_heatmap(df),
                use_container_width=True,
            )

        with col_right:
            st.plotly_chart(
                sub_metering_donut(df),
                use_container_width=True,
            )

        # ── Severity distribution ──────────────────────────────────
        anomalies_df = get_anomalies()
        if not anomalies_df.empty:
            st.plotly_chart(
                severity_distribution(anomalies_df),
                use_container_width=True,
            )
    else:
        st.warning(
            "⚠️ No processed data found. Please run the data processor first.\n\n"
            "```bash\ndocker compose up processor\n```"
        )


def _kpi_card(title: str, value: str, subtitle: str, color: str) -> str:
    """Generate HTML for a KPI card."""
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.95));
        border: 1px solid rgba(148,163,184,0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    ">
        <div style="color: #94A3B8; font-size: 0.8rem; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 8px;">
            {title}
        </div>
        <div style="color: {color}; font-size: 1.8rem; font-weight: 700;
                    margin-bottom: 4px;">
            {value}
        </div>
        <div style="color: #64748B; font-size: 0.75rem;">
            {subtitle}
        </div>
    </div>
    """
