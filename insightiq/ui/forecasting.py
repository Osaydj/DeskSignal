"""Streamlit interface for deterministic sales forecasting."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from insightiq.analytics.forecasting import (
    DailyForecast,
    create_daily_forecast,
)
from insightiq.visualization.forecasting import daily_forecast_figure


FORECAST_HORIZONS = (7, 14, 30, 60, 90)
LOOKBACK_WINDOWS = (30, 60, 90, 180, 365)


def _format_currency(value: float) -> str:
    """Format a number as U.S. currency."""

    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _trend_percentage(result: DailyForecast) -> float:
    """Express the daily slope relative to recent average performance."""

    recent_average = abs(float(result.history["actual"].mean()))

    if recent_average == 0:
        return 0.0

    return (result.slope_per_day / recent_average) * 100


def _render_forecast_metric(
    data: pd.DataFrame,
    *,
    metric: str,
    horizon_days: int,
    lookback_days: int,
) -> None:
    """Render one forecast chart, summary, table, and download."""

    try:
        result = create_daily_forecast(
            data,
            metric=metric,
            horizon_days=horizon_days,
            lookback_days=lookback_days,
        )
    except ValueError as error:
        st.warning(str(error))
        return

    forecast_total = float(result.forecast["forecast"].sum())
    forecast_average = float(result.forecast["forecast"].mean())
    trend_percentage = _trend_percentage(result)

    summary_columns = st.columns(3)

    summary_columns[0].metric(
        f"{horizon_days}-day forecast total",
        _format_currency(forecast_total),
    )
    summary_columns[1].metric(
        "Forecast daily average",
        _format_currency(forecast_average),
    )
    summary_columns[2].metric(
        "Estimated daily trend",
        _format_currency(result.slope_per_day),
        delta=f"{trend_percentage:+.2f}% of recent daily average",
    )

    st.plotly_chart(
        daily_forecast_figure(result),
        width="stretch",
        key=f"{metric}_forecast_chart",
    )

    st.info(
        "This is a deterministic baseline estimate based on recent trends "
        "and day-of-week patterns. Forecasts are uncertain and should not "
        "be treated as guaranteed business results."
    )

    display_table = result.forecast.rename(
        columns={
            "date": "Date",
            "forecast": "Forecast ($)",
            "lower_bound": "Lower estimate ($)",
            "upper_bound": "Upper estimate ($)",
        }
    ).copy()

    display_table["Date"] = pd.to_datetime(
        display_table["Date"]
    ).dt.date

    display_table = display_table.round(
        {
            "Forecast ($)": 2,
            "Lower estimate ($)": 2,
            "Upper estimate ($)": 2,
        }
    )

    with st.expander("View forecast values"):
        st.dataframe(
            display_table,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            label=f"Download {metric} forecast CSV",
            data=result.forecast.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"{metric}_{horizon_days}_day_forecast.csv"
            ),
            mime="text/csv",
            key=f"download_{metric}_forecast",
        )


def render_forecasting(data: pd.DataFrame) -> None:
    """Render revenue and profit forecasting controls."""

    st.subheader("Sales forecasting")
    st.caption(
        "Python estimates future daily revenue and profit using a recent "
        "linear trend, day-of-week patterns, and a statistical uncertainty "
        "range. No AI model or API is used."
    )

    control_columns = st.columns(2)

    with control_columns[0]:
        horizon_days = int(
            st.selectbox(
                "Forecast horizon",
                options=FORECAST_HORIZONS,
                index=2,
                format_func=lambda days: f"{days} days",
                key="forecast_horizon",
            )
        )

    with control_columns[1]:
        lookback_days = int(
            st.selectbox(
                "Historical lookback",
                options=LOOKBACK_WINDOWS,
                index=2,
                format_func=lambda days: f"{days} days",
                key="forecast_lookback",
                help=(
                    "The number of recent calendar days used to estimate "
                    "the trend and weekly pattern."
                ),
            )
        )
    if horizon_days > lookback_days:
        st.warning(
            "The forecast horizon is longer than the historical lookback. "
            "Choose an equal or longer lookback for a more stable estimate. "
            "Short lookbacks may overreact to temporary changes."
        )

    revenue_tab, profit_tab = st.tabs(["Revenue", "Profit"])

    with revenue_tab:
        _render_forecast_metric(
            data,
            metric="revenue",
            horizon_days=horizon_days,
            lookback_days=lookback_days,
        )

    with profit_tab:
        _render_forecast_metric(
            data,
            metric="profit",
            horizon_days=horizon_days,
            lookback_days=lookback_days,
        )