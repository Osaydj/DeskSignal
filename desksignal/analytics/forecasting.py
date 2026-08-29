"""Deterministic daily forecasting for validated sales data."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Final

import pandas as pd


SUPPORTED_METRICS: Final = frozenset({"revenue", "profit"})
MINIMUM_HISTORY_DAYS: Final = 28
MAXIMUM_FORECAST_DAYS: Final = 180


@dataclass(frozen=True)
class DailyForecast:
    """Store historical values, future estimates, and model details."""

    metric: str
    history: pd.DataFrame
    forecast: pd.DataFrame
    horizon_days: int
    slope_per_day: float
    residual_standard_deviation: float


def _validate_settings(
    *,
    metric: str,
    horizon_days: int,
    lookback_days: int,
) -> None:
    """Reject unsupported or unsafe forecasting settings."""

    if metric not in SUPPORTED_METRICS:
        supported = ", ".join(sorted(SUPPORTED_METRICS))
        raise ValueError(f"Metric must be one of: {supported}.")

    if (
        isinstance(horizon_days, bool)
        or not isinstance(horizon_days, int)
        or not 1 <= horizon_days <= MAXIMUM_FORECAST_DAYS
    ):
        raise ValueError(
            f"horizon_days must be between 1 and {MAXIMUM_FORECAST_DAYS}."
        )

    if (
        isinstance(lookback_days, bool)
        or not isinstance(lookback_days, int)
        or lookback_days < MINIMUM_HISTORY_DAYS
    ):
        raise ValueError(
            f"lookback_days must be at least {MINIMUM_HISTORY_DAYS}."
        )


def create_daily_forecast(
    data: pd.DataFrame,
    *,
    metric: str = "revenue",
    horizon_days: int = 30,
    lookback_days: int = 90,
) -> DailyForecast:
    """Create a deterministic daily forecast.

    The model uses:

    1. A linear trend calculated from recent daily totals.
    2. Additive day-of-week adjustments.
    3. A statistical uncertainty range based on historical model errors.

    This is an educational baseline forecast. It should support investigation,
    not be treated as a guaranteed prediction.
    """

    _validate_settings(
        metric=metric,
        horizon_days=horizon_days,
        lookback_days=lookback_days,
    )

    required_columns = {"date", metric}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}.")

    if data.empty:
        raise ValueError("Forecasting requires a non-empty dataset.")

    cleaned = data.loc[:, ["date", metric]].copy()
    cleaned["date"] = pd.to_datetime(
        cleaned["date"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)
    cleaned[metric] = pd.to_numeric(cleaned[metric], errors="coerce")

    if cleaned["date"].isna().any():
        raise ValueError("The date column contains invalid values.")

    if cleaned[metric].isna().any():
        raise ValueError(f"The {metric} column contains invalid values.")

    cleaned["date"] = cleaned["date"].dt.normalize()

    daily_values = (
        cleaned.groupby("date", as_index=True)[metric]
        .sum()
        .sort_index()
        .astype(float)
    )

    calendar = pd.date_range(
        start=daily_values.index.min(),
        end=daily_values.index.max(),
        freq="D",
    )
    daily_values = daily_values.reindex(calendar, fill_value=0.0)

    if len(daily_values) < MINIMUM_HISTORY_DAYS:
        raise ValueError(
            "Forecasting requires at least "
            f"{MINIMUM_HISTORY_DAYS} calendar days of history."
        )

    training = daily_values.tail(
        min(lookback_days, len(daily_values))
    ).copy()
    training_size = len(training)

    time_index = pd.Series(
        range(training_size),
        index=training.index,
        dtype=float,
    )

    mean_time = float(time_index.mean())
    mean_value = float(training.mean())

    denominator = float(((time_index - mean_time) ** 2).sum())

    if denominator == 0:
        slope = 0.0
    else:
        slope = float(
            (
                (time_index - mean_time)
                * (training - mean_value)
            ).sum()
            / denominator
        )

    intercept = mean_value - (slope * mean_time)
    trend_values = intercept + (slope * time_index)

    initial_residuals = training - trend_values
    weekday_effects = (
        initial_residuals.groupby(training.index.dayofweek)
        .mean()
        .to_dict()
    )

    fitted_values = pd.Series(
        [
            (
                intercept
                + (slope * position)
                + float(weekday_effects.get(date.dayofweek, 0.0))
            )
            for position, date in enumerate(training.index)
        ],
        index=training.index,
        dtype=float,
    )

    if metric == "revenue":
        fitted_values = fitted_values.clip(lower=0.0)

    model_errors = training - fitted_values
    residual_standard_deviation = float(model_errors.std(ddof=1))

    if pd.isna(residual_standard_deviation):
        residual_standard_deviation = 0.0

    future_dates = pd.date_range(
        start=training.index.max() + pd.DateOffset(days=1),
        periods=horizon_days,
        freq="D",
    )

    forecast_rows: list[dict[str, object]] = []

    for step, future_date in enumerate(future_dates, start=1):
        future_position = (training_size - 1) + step
        weekday_effect = float(
            weekday_effects.get(future_date.dayofweek, 0.0)
        )

        predicted_value = (
            intercept
            + (slope * future_position)
            + weekday_effect
        )

        if metric == "revenue":
            predicted_value = max(0.0, predicted_value)

        uncertainty = (
            1.96
            * residual_standard_deviation
            * sqrt(1 + (step / training_size))
        )

        lower_bound = predicted_value - uncertainty
        upper_bound = predicted_value + uncertainty

        if metric == "revenue":
            lower_bound = max(0.0, lower_bound)

        forecast_rows.append(
            {
                "date": future_date,
                "forecast": float(predicted_value),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }
        )

    history_frame = (
        training.rename("actual")
        .rename_axis("date")
        .reset_index()
    )
    forecast_frame = pd.DataFrame(forecast_rows)

    return DailyForecast(
        metric=metric,
        history=history_frame,
        forecast=forecast_frame,
        horizon_days=horizon_days,
        slope_per_day=slope,
        residual_standard_deviation=residual_standard_deviation,
    )