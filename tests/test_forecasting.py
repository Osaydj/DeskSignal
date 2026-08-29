"""Tests for deterministic daily forecasting."""

from __future__ import annotations

import pandas as pd
import pytest

from insightiq.analytics.forecasting import create_daily_forecast
from insightiq.visualization.forecasting import daily_forecast_figure

def _sample_history(days: int = 120) -> pd.DataFrame:
    """Create predictable fictional daily results for testing."""

    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    revenue = [
        1_000.0 + (day * 5.0) + ((day % 7) * 20.0)
        for day in range(days)
    ]

    return pd.DataFrame(
        {
            "date": dates,
            "revenue": revenue,
            "profit": [value * 0.4 for value in revenue],
        }
    )


def test_revenue_forecast_has_expected_structure() -> None:
    data = _sample_history()

    result = create_daily_forecast(
        data,
        metric="revenue",
        horizon_days=30,
        lookback_days=90,
    )

    assert result.metric == "revenue"
    assert result.horizon_days == 30
    assert len(result.history) == 90
    assert len(result.forecast) == 30
    assert list(result.history.columns) == ["date", "actual"]
    assert list(result.forecast.columns) == [
        "date",
        "forecast",
        "lower_bound",
        "upper_bound",
    ]
    assert result.forecast.iloc[0]["date"] == (
        data["date"].max() + pd.DateOffset(days=1)
    )


def test_revenue_forecast_values_and_bounds_are_valid() -> None:
    result = create_daily_forecast(_sample_history())

    assert (result.forecast["forecast"] >= 0).all()
    assert (result.forecast["lower_bound"] >= 0).all()
    assert (
        result.forecast["lower_bound"]
        <= result.forecast["forecast"]
    ).all()
    assert (
        result.forecast["forecast"]
        <= result.forecast["upper_bound"]
    ).all()
    assert result.slope_per_day > 0


def test_forecast_is_deterministic() -> None:
    data = _sample_history()

    first_result = create_daily_forecast(data)
    second_result = create_daily_forecast(data)

    pd.testing.assert_frame_equal(
        first_result.history,
        second_result.history,
    )
    pd.testing.assert_frame_equal(
        first_result.forecast,
        second_result.forecast,
    )
    assert first_result.slope_per_day == second_result.slope_per_day


def test_profit_forecast_is_supported() -> None:
    result = create_daily_forecast(
        _sample_history(),
        metric="profit",
        horizon_days=14,
    )

    assert result.metric == "profit"
    assert len(result.forecast) == 14


def test_missing_calendar_days_are_filled_with_zero() -> None:
    data = _sample_history()
    missing_date = data.loc[50, "date"]
    data = data.drop(index=50)

    result = create_daily_forecast(
        data,
        lookback_days=90,
    )

    missing_day = result.history.loc[
        result.history["date"] == missing_date,
        "actual",
    ]

    assert len(missing_day) == 1
    assert missing_day.iloc[0] == 0.0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"metric": "cost"}, "Metric must be one of"),
        ({"horizon_days": 0}, "horizon_days"),
        ({"horizon_days": 181}, "horizon_days"),
        ({"horizon_days": True}, "horizon_days"),
        ({"lookback_days": 27}, "lookback_days"),
    ],
)
def test_invalid_settings_are_rejected(
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_daily_forecast(_sample_history(), **options)


def test_missing_required_column_is_rejected() -> None:
    data = _sample_history().drop(columns="revenue")

    with pytest.raises(ValueError, match="Missing required columns"):
        create_daily_forecast(data)


def test_insufficient_history_is_rejected() -> None:
    data = _sample_history(days=20)

    with pytest.raises(
        ValueError,
        match="at least 28 calendar days",
    ):
        create_daily_forecast(data)


def test_forecast_chart_contains_history_bounds_and_forecast() -> None:
    result = create_daily_forecast(
        _sample_history(),
        metric="revenue",
        horizon_days=30,
    )

    figure = daily_forecast_figure(result)

    assert len(figure.data) == 4
    assert [trace.name for trace in figure.data] == [
        "Actual revenue",
        "Upper bound",
        "95% uncertainty range",
        "Forecast revenue",
    ]
    assert len(figure.layout.shapes) == 1
    assert figure.layout.title.text == "30-day revenue forecast"