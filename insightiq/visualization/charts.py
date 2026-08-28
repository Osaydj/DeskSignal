"""Interactive Plotly chart builders for validated sales data."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from insightiq.analytics.performance import (
    category_performance,
    daily_performance,
    product_performance,
    region_performance,
)


REVENUE_COLOR = "#2563EB"
PROFIT_COLOR = "#0F766E"
NEGATIVE_COLOR = "#B91C1C"
GRID_COLOR = "rgba(128, 128, 128, 0.18)"


def _style_figure(
    figure: go.Figure,
    *,
    height: int = 380,
    currency_axis: str = "y",
) -> go.Figure:
    """Apply a consistent, restrained dashboard style."""

    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        hoverlabel={"namelength": -1},
        legend_title_text="",
        font={"family": "Arial, sans-serif"},
    )
    figure.update_xaxes(gridcolor=GRID_COLOR)
    figure.update_yaxes(gridcolor=GRID_COLOR)
    if currency_axis == "x":
        figure.update_xaxes(tickprefix="$", separatethousands=True)
    elif currency_axis == "y":
        figure.update_yaxes(tickprefix="$", separatethousands=True)
    return figure


def revenue_over_time_figure(data: pd.DataFrame) -> go.Figure:
    """Create an interactive daily revenue line chart."""

    daily = daily_performance(data)
    figure = px.line(
        daily,
        x="date",
        y="revenue",
        title="Revenue over time",
        labels={"date": "Date", "revenue": "Revenue"},
    )
    figure.update_traces(
        line={"color": REVENUE_COLOR, "width": 2.5},
        hovertemplate="%{x|%b %d, %Y}<br>Revenue: $%{y:,.2f}<extra></extra>",
    )
    figure.update_layout(hovermode="x unified")
    return _style_figure(figure)


def profit_over_time_figure(data: pd.DataFrame) -> go.Figure:
    """Create an interactive daily profit line chart."""

    daily = daily_performance(data)
    figure = px.line(
        daily,
        x="date",
        y="profit",
        title="Profit over time",
        labels={"date": "Date", "profit": "Profit"},
    )
    figure.update_traces(
        line={"color": PROFIT_COLOR, "width": 2.5},
        hovertemplate="%{x|%b %d, %Y}<br>Profit: $%{y:,.2f}<extra></extra>",
    )
    figure.update_layout(hovermode="x unified")
    return _style_figure(figure)


def revenue_by_category_figure(data: pd.DataFrame) -> go.Figure:
    """Create a revenue-by-category bar chart."""

    summary = category_performance(data)
    figure = px.bar(
        summary,
        x="category",
        y="revenue",
        title="Revenue by category",
        labels={"category": "Category", "revenue": "Revenue"},
    )
    figure.update_traces(
        marker_color=REVENUE_COLOR,
        hovertemplate="%{x}<br>Revenue: $%{y:,.2f}<extra></extra>",
    )
    figure.update_xaxes(showgrid=False)
    return _style_figure(figure)


def profit_by_region_figure(data: pd.DataFrame) -> go.Figure:
    """Create a profit-by-region chart that also supports negative values."""

    summary = region_performance(data)
    colors = [PROFIT_COLOR if value >= 0 else NEGATIVE_COLOR for value in summary["profit"]]
    figure = go.Figure(
        data=[
            go.Bar(
                x=summary["region"],
                y=summary["profit"],
                marker_color=colors,
                hovertemplate="%{x}<br>Profit: $%{y:,.2f}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title="Profit by region",
        xaxis_title="Region",
        yaxis_title="Profit",
    )
    figure.update_xaxes(showgrid=False)
    return _style_figure(figure)


def top_products_figure(data: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Create a horizontal chart of products ranked by revenue."""

    summary = product_performance(data, top_n=top_n).sort_values(
        "revenue", ascending=True
    )
    figure = px.bar(
        summary,
        x="revenue",
        y="product",
        orientation="h",
        title=f"Top {min(top_n, len(summary))} products by revenue",
        labels={"product": "Product", "revenue": "Revenue"},
    )
    figure.update_traces(
        marker_color=REVENUE_COLOR,
        customdata=summary[["units_sold", "profit"]],
        hovertemplate=(
            "%{y}<br>Revenue: $%{x:,.2f}<br>"
            "Units: %{customdata[0]:,.0f}<br>"
            "Profit: $%{customdata[1]:,.2f}<extra></extra>"
        ),
    )
    figure.update_yaxes(title=None)
    return _style_figure(figure, height=440, currency_axis="x")
