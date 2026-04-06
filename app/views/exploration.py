"""
Exploration page — Interactive data drill-down.

Provides zoomable time-series charts with anomaly overlay
highlights, sub-meter comparison, and raw data table access.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.charts import COLORS, _apply_theme, consumption_trend
from app.components.filters import date_range_filter, sub_meter_selector
from app.services.data_service import load_anomaly_scores, load_consumption_data


def render() -> None:
    """Render the Exploration page."""
    st.markdown(
        """
        <h1 style="
            background: linear-gradient(135deg, #06B6D4, #10B981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        ">🔬 Data Exploration</h1>
        <p style="color: #94A3B8; margin-bottom: 1.5rem;">
            Interactive drill-down into energy consumption data with anomaly overlays.
        </p>
        """,
        unsafe_allow_html=True,
    )

    df = load_consumption_data()
    scores = load_anomaly_scores()

    if df.empty:
        st.warning("No processed data available. Run the processor first.")
        return

    # ── Filters ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛️ Exploration Controls")
        start_date, end_date = date_range_filter(df, key_prefix="explore")
        sub_meters = sub_meter_selector(key="explore_submeters")

        show_anomalies = st.checkbox("Show Anomaly Markers", value=True, key="explore_anomalies")
        chart_type = st.radio(
            "Chart Type",
            ["Line", "Area"],
            index=0,
            key="explore_chart_type",
        )

    # ── Filter data ────────────────────────────────────────────────
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_filtered = df.loc[mask]

    if df_filtered.empty:
        st.info("No data in the selected date range.")
        return

    st.markdown(f"**Showing {len(df_filtered):,} data points** "
                f"from {start_date} to {end_date}")

    # ── Main consumption chart ─────────────────────────────────────
    anomaly_overlay = None
    if show_anomalies and not scores.empty:
        scores_filtered = scores.loc[scores.index.isin(df_filtered.index)]
        if "is_anomaly" in scores_filtered.columns:
            anomaly_overlay = scores_filtered["is_anomaly"]

    fig = consumption_trend(
        df_filtered,
        title="Global Active Power",
        anomaly_scores=anomaly_overlay,
    )

    if chart_type == "Area":
        fig.data[0].update(fill="tozeroy", fillcolor="rgba(99,102,241,0.15)")

    st.plotly_chart(fig, use_container_width=True, key="explore_main")

    # ── Sub-meter comparison ───────────────────────────────────────
    if sub_meters:
        st.markdown("### ⚡ Sub-Meter Comparison")

        available_cols = [c for c in sub_meters if c in df_filtered.columns]
        if available_cols:
            fig_sub = go.Figure()
            meter_colors = {
                "Sub_metering_1": COLORS["primary"],
                "Sub_metering_2": COLORS["accent"],
                "Sub_metering_3": COLORS["warning"],
                "Unmetered_consumption": COLORS["muted"],
            }
            meter_labels = {
                "Sub_metering_1": "Kitchen",
                "Sub_metering_2": "Laundry",
                "Sub_metering_3": "Water Heater & AC",
                "Unmetered_consumption": "Other",
            }

            for col in available_cols:
                fig_sub.add_trace(go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered[col],
                    mode="lines",
                    name=meter_labels.get(col, col),
                    line=dict(color=meter_colors.get(col, COLORS["text"]), width=1.5),
                ))

            fig_sub.update_layout(
                title="Sub-Meter Energy Consumption (Wh)",
                xaxis_title="Time",
                yaxis_title="Energy (Wh)",
            )
            st.plotly_chart(
                _apply_theme(fig_sub),
                use_container_width=True,
                key="explore_submeters_chart",
            )

    # ── Voltage stability chart ────────────────────────────────────
    if "Voltage" in df_filtered.columns:
        with st.expander("🔌 Voltage Stability", expanded=False):
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(
                x=df_filtered.index,
                y=df_filtered["Voltage"],
                mode="lines",
                name="Voltage (V)",
                line=dict(color=COLORS["success"], width=1),
            ))
            fig_v.update_layout(
                title="Voltage Over Time",
                xaxis_title="Time",
                yaxis_title="Voltage (V)",
            )
            st.plotly_chart(
                _apply_theme(fig_v),
                use_container_width=True,
                key="explore_voltage",
            )

    # ── Raw data table ─────────────────────────────────────────────
    with st.expander("📋 Raw Data Table", expanded=False):
        st.dataframe(
            df_filtered.head(1000).style.format("{:.3f}"),
            use_container_width=True,
            height=400,
        )
        st.caption("Showing first 1,000 rows. Full dataset available via Parquet export.")

        # Download button
        csv = df_filtered.to_csv()
        st.download_button(
            "📥 Download filtered data (CSV)",
            csv,
            file_name="energy_data_filtered.csv",
            mime="text/csv",
            key="explore_download",
        )
