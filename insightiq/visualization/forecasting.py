
"""Plotly visualizations for deterministic sales forecasts."""

from __future__ import annotations

import plotly.graph_objects as go

from insightiq.analytics.forecasting import DailyForecast


REVENUE_COLOR = "#2563EB"
PROFIT_COLOR = "#0F8B8D"
FORECAST_COLOR = "#F59E0B"
UNCERTAINTY_COLOR = "rgba(245, 158, 11, 0.18)"


def daily_forecast_figure(result: DailyForecast) -> go.Figure:
    """Plot historical results, forecasts, and uncertainty bounds."""

    if result.history.empty:
        raise ValueError("Forecast history cannot be empty.")

    if result.forecast.empty:
        raise ValueError("Forecast results cannot be empty.")

    display_name = result.metric.replace("_", " ").title()
    actual_color = (
        REVENUE_COLOR
        if result.metric == "revenue"
        else PROFIT_COLOR
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=result.history["date"],
            y=result.history["actual"],
            mode="lines",
            name=f"Actual {display_name.lower()}",
            line={
                "color": actual_color,
                "width": 2.5,
            },
            hovertemplate=(
                "%{x|%b %d, %Y}<br>"
                "Actual: $%{y:,.2f}<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=result.forecast["date"],
            y=result.forecast["upper_bound"],
            mode="lines",
            name="Upper bound",
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
            name="95% uncertainty range",
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
            name=f"Forecast {display_name.lower()}",
            line={
                "color": FORECAST_COLOR,
                "width": 2.5,
                "dash": "dash",
            },
            marker={
                "color": FORECAST_COLOR,
                "size": 5,
            },
            customdata=result.forecast[
                ["lower_bound", "upper_bound"]
            ],
            hovertemplate=(
                "%{x|%b %d, %Y}<br>"
                "Forecast: $%{y:,.2f}<br>"
                "Lower estimate: $%{customdata[0]:,.2f}<br>"
                "Upper estimate: $%{customdata[1]:,.2f}"
                "<extra></extra>"
            ),
        )
    )

    forecast_start = result.forecast.iloc[0]["date"]

    figure.add_vline(
        x=forecast_start,
        line_width=1.5,
        line_dash="dot",
        line_color="#9CA3AF",
    )

    figure.update_layout(
        title=(
            f"{result.horizon_days}-day "
            f"{display_name.lower()} forecast"
        ),
        xaxis_title="Date",
        yaxis_title=f"{display_name} ($)",
        hovermode="x unified",
        height=440,
        margin={
            "l": 40,
            "r": 30,
            "t": 60,
            "b": 40,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )

    return figure