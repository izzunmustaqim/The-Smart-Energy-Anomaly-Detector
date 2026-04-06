"""
Smart Alert card component for the Streamlit UI.

Renders a single anomaly as a styled card with severity badge,
metric columns, contextual explanation, and a mini context chart.
"""

import pandas as pd
import streamlit as st

from app.components.charts import anomaly_context_chart


def render_alert_card(
    alert: pd.Series | dict,
    consumption_df: pd.DataFrame,
    index: int = 0,
) -> None:
    """
    Render a single anomaly alert card.

    Parameters
    ----------
    alert : pd.Series or dict
        A row from the anomalies table.
    consumption_df : pd.DataFrame
        Full consumption DataFrame for context chart rendering.
    index : int
        Card index for unique Streamlit keys.
    """
    severity = alert.get("severity", "low")
    emoji = alert.get("severity_emoji", "⚪")
    timestamp = alert.get("timestamp", "N/A")
    actual = alert.get("actual_value", 0)
    expected_mean = alert.get("expected_mean", 0)
    expected_min = alert.get("expected_min", 0)
    expected_max = alert.get("expected_max", 0)
    deviation_pct = alert.get("deviation_pct", 0)
    direction = alert.get("direction", "spike")
    score = alert.get("ensemble_score", 0)
    explanation = alert.get("human_readable_text", "No explanation available.")
    factors = alert.get("contributing_factors", [])

    # Severity-based border color
    severity_border = {
        "critical": "#EF4444",
        "high": "#F97316",
        "medium": "#F59E0B",
        "low": "#10B981",
    }
    border_color = severity_border.get(severity, "#94A3B8")

    # ── Card container ─────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {border_color};
            background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
            backdrop-filter: blur(10px);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="
                    background: {border_color}22;
                    color: {border_color};
                    padding: 4px 12px;
                    border-radius: 100px;
                    font-size: 0.8rem;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                ">{emoji} {severity}</span>
                <span style="color: #94A3B8; font-size: 0.85rem;">
                    Score: {score:.3f}
                </span>
            </div>
            <div style="color: #E2E8F0; font-size: 0.95rem; font-weight: 500; margin-bottom: 8px;">
                📅 {timestamp}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metrics row ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Actual Power",
            f"{actual:.2f} kW",
            f"{'+' if deviation_pct > 0 else ''}{deviation_pct:.1f}%",
            delta_color="inverse" if direction == "spike" else "normal",
        )
    with col2:
        st.metric("Expected Mean", f"{expected_mean:.2f} kW")
    with col3:
        st.metric("Expected Range", f"{expected_min:.2f}–{expected_max:.2f} kW")
    with col4:
        st.metric("Direction", f"{'📈' if direction == 'spike' else '📉'} {direction.title()}")

    # ── Explanation ────────────────────────────────────────────────
    with st.expander(f"🔍 Detailed Explanation", expanded=(index < 3)):
        st.markdown(
            f"""<div style="
                background: rgba(99,102,241,0.05);
                border-radius: 8px;
                padding: 16px;
                font-size: 0.9rem;
                line-height: 1.6;
                color: #CBD5E1;
                white-space: pre-wrap;
            ">{explanation}</div>""",
            unsafe_allow_html=True,
        )

        # Contributing factors as badges
        if factors and isinstance(factors, list) and len(factors) > 0:
            st.markdown("**Contributing Factors:**")
            for factor in factors:
                st.markdown(f"- {factor}")

    # ── Context mini-chart ─────────────────────────────────────────
    if not consumption_df.empty and isinstance(timestamp, pd.Timestamp):
        with st.expander("📊 Context Chart (±24h)", expanded=False):
            fig = anomaly_context_chart(consumption_df, timestamp)
            st.plotly_chart(fig, use_container_width=True, key=f"ctx_{index}")

    st.markdown("---")
