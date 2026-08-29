"""Plotly visualizations for DeskSignal's local machine-learning models."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from desksignal.analytics.machine_learning import MLForecast


ML_COLOR = "#7C3AED"
REVENUE_COLOR = "#2563EB"
PROFIT_COLOR = "#0F766E"
FORECAST_COLOR = "#F59E0B"
GRID_COLOR = "rgba(128, 128, 128, 0.18)"
UNCERTAINTY_COLOR = "rgba(245, 158, 11, 0.18)"


def _style_figure(figure: go.Figure, *, height: int = 440) -> go.Figure:
    """Apply the shared, restrained dashboard chart style."""

    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 30, "r": 30, "t": 60, "b": 35},
        hoverlabel={"namelength": -1},
        legend_title_text="",
        font={"family": "Arial, sans-serif"},
    )
    figure.update_xaxes(gridcolor=GRID_COLOR)
    figure.update_yaxes(gridcolor=GRID_COLOR)
    return figure


def ml_anomaly_figure(results: pd.DataFrame, *, metric: str) -> go.Figure:
    """Plot daily results and Isolation Forest anomaly flags."""

    if metric not in {"revenue", "profit"}:
        raise ValueError("metric must be revenue or profit.")
    if results.empty:
        raise ValueError("ML anomaly results cannot be empty.")

    display_name = metric.title()
    actual_color = REVENUE_COLOR if metric == "revenue" else PROFIT_COLOR
    anomalies = results.loc[results["is_anomaly"]].copy()

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=results["date"],
            y=results[metric],
            mode="lines",
            name=f"Daily {metric}",
            line={"color": actual_color, "width": 2.2},
            hovertemplate=(
                "%{x|%b %d, %Y}<br>"
                f"{display_name}: $%{{y:,.2f}}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=anomalies["date"],
            y=anomalies[metric],
            mode="markers",
            name="ML-flagged day",
            marker={
                "color": ML_COLOR,
                "size": 11,
                "symbol": "diamond",
                "line": {"color": "white", "width": 1},
            },
            customdata=anomalies[["anomaly_score", "direction"]],
            hovertemplate=(
                "%{x|%b %d, %Y}<br>"
                f"{display_name}: $%{{y:,.2f}}<br>"
                "Anomaly score: %{customdata[0]:.1f}/100<br>"
                "Direction: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"Isolation Forest: daily {metric}",
        xaxis_title="Date",
        yaxis_title=f"{display_name} ($)",
        hovermode="x unified",
    )
    figure.update_yaxes(tickprefix="$", separatethousands=True)
    return _style_figure(figure)


def market_segments_figure(assignments: pd.DataFrame) -> go.Figure:
    """Plot K-Means market groups by revenue, profit, and transaction volume."""

    if assignments.empty:
        raise ValueError("Market-segment assignments cannot be empty.")

    chart_data = assignments.copy()
    chart_data["cluster_label"] = chart_data["cluster"].map(
        lambda value: f"Cluster {int(value)}"
    )
    figure = px.scatter(
        chart_data,
        x="revenue",
        y="profit",
        color="cluster_label",
        size="transactions",
        size_max=32,
        hover_data={
            "product": True,
            "category": True,
            "region": True,
            "customer_type": True,
            "profile": True,
            "cluster_label": False,
            "transactions": ":,.0f",
            "revenue": ":$,.2f",
            "profit": ":$,.2f",
        },
        labels={
            "revenue": "Revenue ($)",
            "profit": "Profit ($)",
            "cluster_label": "Cluster",
        },
        title="K-Means product-market segments",
        color_discrete_sequence=[
            "#7C3AED",
            "#2563EB",
            "#0F766E",
            "#D97706",
            "#DB2777",
            "#4B5563",
            "#0891B2",
            "#65A30D",
        ],
    )
    figure.update_layout(legend={"orientation": "h", "y": 1.12})
    figure.update_xaxes(tickprefix="$", separatethousands=True)
    figure.update_yaxes(tickprefix="$", separatethousands=True)
    return _style_figure(figure, height=480)


def ml_forecast_figure(result: MLForecast) -> go.Figure:
    """Plot actual history and a Random Forest future forecast."""

    if result.history.empty or result.forecast.empty:
        raise ValueError("ML forecast history and results cannot be empty.")

    display_name = result.metric.title()
    actual_color = REVENUE_COLOR if result.metric == "revenue" else PROFIT_COLOR
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.history["date"],
            y=result.history["actual"],
            mode="lines",
            name=f"Actual {result.metric}",
            line={"color": actual_color, "width": 2.2},
            hovertemplate=(
                "%{x|%b %d, %Y}<br>Actual: $%{y:,.2f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.forecast["date"],
            y=result.forecast["upper_bound"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.forecast["date"],
            y=result.forecast["lower_bound"],
            mode="lines",
            name="Validation-error range",
            line={"width": 0},
            fill="tonexty",
            fillcolor=UNCERTAINTY_COLOR,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.forecast["date"],
            y=result.forecast["forecast"],
            mode="lines+markers",
            name="Random Forest forecast",
            line={"color": FORECAST_COLOR, "width": 2.5, "dash": "dash"},
            marker={"color": FORECAST_COLOR, "size": 5},
            customdata=result.forecast[["lower_bound", "upper_bound"]],
            hovertemplate=(
                "%{x|%b %d, %Y}<br>Forecast: $%{y:,.2f}<br>"
                "Lower estimate: $%{customdata[0]:,.2f}<br>"
                "Upper estimate: $%{customdata[1]:,.2f}<extra></extra>"
            ),
        )
    )
    figure.add_vline(
        x=result.forecast.iloc[0]["date"],
        line_width=1.5,
        line_dash="dot",
        line_color="#9CA3AF",
    )
    figure.update_layout(
        title=f"Random Forest: {result.horizon_days}-day {result.metric} forecast",
        xaxis_title="Date",
        yaxis_title=f"{display_name} ($)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
    )
    figure.update_yaxes(tickprefix="$", separatethousands=True)
    return _style_figure(figure)
