"""
Plotly chart builders with consistent dark theme styling.

All chart functions return a plotly.graph_objects.Figure that
can be rendered directly with st.plotly_chart().
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Shared theme ───────────────────────────────────────────────────
COLORS = {
    "primary": "#6366F1",       # Indigo
    "secondary": "#8B5CF6",     # Violet
    "accent": "#06B6D4",        # Cyan
    "success": "#10B981",       # Emerald
    "warning": "#F59E0B",       # Amber
    "danger": "#EF4444",        # Red
    "bg": "#0F172A",            # Slate 900
    "surface": "#1E293B",       # Slate 800
    "text": "#E2E8F0",          # Slate 200
    "muted": "#94A3B8",         # Slate 400
}

SEVERITY_COLORS = {
    "critical": "#EF4444",
    "high": "#F97316",
    "medium": "#F59E0B",
    "low": "#10B981",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    hovermode="x unified",
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    ),
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the shared dark theme to a figure."""
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.1)",
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.1)",
        zeroline=False,
    )
    return fig


# ── Chart builders ─────────────────────────────────────────────────

def consumption_trend(
    df: pd.DataFrame,
    column: str = "Global_active_power",
    title: str = "Energy Consumption Trend",
    anomaly_scores: pd.Series | None = None,
) -> go.Figure:
    """
    Line chart of power consumption over time, optionally with
    anomaly highlight markers.
    """
    fig = go.Figure()

    # Main trend line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[column],
        mode="lines",
        name="Power (kW)",
        line=dict(color=COLORS["primary"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.1)",
    ))

    # Anomaly markers
    if anomaly_scores is not None:
        anomaly_mask = anomaly_scores > 0
        if anomaly_mask.any():
            anomaly_data = df.loc[anomaly_mask]
            fig.add_trace(go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data[column],
                mode="markers",
                name="Anomaly",
                marker=dict(
                    color=COLORS["danger"],
                    size=6,
                    symbol="diamond",
                    line=dict(width=1, color="white"),
                ),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Power (kW)",
    )
    return _apply_theme(fig)


def hourly_heatmap(
    df: pd.DataFrame,
    column: str = "Global_active_power",
    title: str = "Consumption Heatmap (Hour × Day of Week)",
) -> go.Figure:
    """
    Heatmap showing average consumption by hour and day of week.
    """
    df_temp = df.copy()
    df_temp["hour"] = df_temp.index.hour
    df_temp["dow"] = df_temp.index.dayofweek

    pivot = df_temp.pivot_table(
        values=column, index="dow", columns="hour", aggfunc="mean"
    )

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=day_labels,
        colorscale=[
            [0, "#0F172A"],
            [0.25, "#312E81"],
            [0.5, "#6366F1"],
            [0.75, "#A78BFA"],
            [1, "#F59E0B"],
        ],
        colorbar=dict(title="kW"),
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Avg Power: %{z:.2f} kW<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
    )
    return _apply_theme(fig)


def sub_metering_donut(
    df: pd.DataFrame,
    title: str = "Energy Distribution by Sub-Meter",
) -> go.Figure:
    """
    Donut chart showing the proportion of energy from each sub-meter.
    """
    labels = ["Kitchen", "Laundry", "Water Heater & AC", "Other"]
    values = [
        df["Sub_metering_1"].sum() if "Sub_metering_1" in df.columns else 0,
        df["Sub_metering_2"].sum() if "Sub_metering_2" in df.columns else 0,
        df["Sub_metering_3"].sum() if "Sub_metering_3" in df.columns else 0,
        df["Unmetered_consumption"].sum() if "Unmetered_consumption" in df.columns else 0,
    ]
    colors = [COLORS["primary"], COLORS["accent"], COLORS["warning"], COLORS["muted"]]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="%{label}: %{value:.0f} Wh (%{percent})<extra></extra>",
    ))

    fig.update_layout(title=title)
    return _apply_theme(fig)


def anomaly_context_chart(
    df: pd.DataFrame,
    anomaly_ts: pd.Timestamp,
    column: str = "Global_active_power",
    context_hours: int = 48,
) -> go.Figure:
    """
    Mini line chart centered on an anomaly showing the surrounding
    context window for the Smart Alerts detail view.
    """
    start = anomaly_ts - pd.Timedelta(hours=context_hours // 2)
    end = anomaly_ts + pd.Timedelta(hours=context_hours // 2)
    window = df.loc[start:end]

    if window.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=window.index,
        y=window[column],
        mode="lines",
        name="Power",
        line=dict(color=COLORS["primary"], width=2),
    ))

    # Highlight anomaly point
    if anomaly_ts in df.index:
        fig.add_trace(go.Scatter(
            x=[anomaly_ts],
            y=[df.loc[anomaly_ts, column]],
            mode="markers",
            name="Anomaly",
            marker=dict(
                color=COLORS["danger"],
                size=14,
                symbol="diamond",
                line=dict(width=2, color="white"),
            ),
        ))

    # Add expected range shading
    dow = anomaly_ts.dayofweek
    hour = anomaly_ts.hour
    mask = (df.index.dayofweek == dow) & (df.index.hour == hour)
    if mask.sum() > 0:
        q10 = df.loc[mask, column].quantile(0.10)
        q90 = df.loc[mask, column].quantile(0.90)
        fig.add_hrect(
            y0=q10, y1=q90,
            fillcolor=COLORS["success"],
            opacity=0.08,
            layer="below",
            line_width=0,
            annotation_text="Expected range",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color=COLORS["muted"],
        )

    fig.update_layout(
        title=f"Context: ±{context_hours // 2}h around anomaly",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        height=250,
        margin=dict(l=30, r=10, t=35, b=30),
    )
    return _apply_theme(fig)


def severity_distribution(
    anomalies_df: pd.DataFrame,
    title: str = "Anomaly Severity Distribution",
) -> go.Figure:
    """Bar chart of anomaly counts by severity."""
    if anomalies_df.empty or "severity" not in anomalies_df.columns:
        return _apply_theme(go.Figure())

    counts = anomalies_df["severity"].value_counts().reindex(
        ["critical", "high", "medium", "low"], fill_value=0
    )

    fig = go.Figure(go.Bar(
        x=counts.index.str.capitalize(),
        y=counts.values,
        marker=dict(
            color=[SEVERITY_COLORS.get(s, COLORS["muted"]) for s in counts.index],
            line=dict(width=0),
        ),
        text=counts.values,
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Severity",
        yaxis_title="Count",
    )
    return _apply_theme(fig)
