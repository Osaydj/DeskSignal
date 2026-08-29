"""Deterministic anomaly detection for daily business metrics."""

from __future__ import annotations

import pandas as pd


SUPPORTED_METRICS = frozenset({"revenue", "profit"})


def detect_daily_anomalies(
    data: pd.DataFrame,
    *,
    metric: str = "revenue",
    window: int = 28,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Compare daily totals with a rolling median and flag unusual values.

    The returned table contains every calendar date in the input. An anomaly
    score is calculated with a robust rolling scale so ordinary seasonal
    movement has less influence than it would with a global average.
    """

    if metric not in SUPPORTED_METRICS:
        supported = ", ".join(sorted(SUPPORTED_METRICS))
        raise ValueError(f"metric must be one of: {supported}.")
    if window < 3:
        raise ValueError("window must be at least 3 days.")
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero.")

    required_columns = {"date", metric}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}.")

    daily = data.loc[:, ["date", metric]].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily[metric] = pd.to_numeric(daily[metric], errors="coerce")
    daily = daily.dropna(subset=["date", metric])

    if daily.empty:
        return _empty_anomaly_table()

    daily = (
        daily.groupby("date", as_index=False, observed=True)[metric]
        .sum()
        .sort_values("date", ignore_index=True)
        .rename(columns={metric: "actual"})
    )

    minimum_observations = min(len(daily), max(3, window // 2))
    actual = daily["actual"]
    baseline = actual.rolling(
        window=window,
        min_periods=minimum_observations,
        center=True,
    ).median()

    difference = actual - baseline
    absolute_deviation = difference.abs()
    median_absolute_deviation = absolute_deviation.rolling(
        window=window,
        min_periods=minimum_observations,
        center=True,
    ).median()

    robust_scale = median_absolute_deviation * 1.4826
    fallback_scale = actual.rolling(
        window=window,
        min_periods=minimum_observations,
        center=True,
    ).std(ddof=0)
    scale = robust_scale.where(robust_scale > 0, fallback_scale)
    anomaly_score = difference.divide(scale.where(scale > 0)).fillna(0.0)

    daily["metric"] = metric
    daily["baseline"] = baseline.fillna(actual.median())
    daily["difference"] = daily["actual"] - daily["baseline"]
    non_zero_baseline = daily["baseline"].abs().where(daily["baseline"] != 0)
    daily["percent_difference"] = (
        daily["difference"].divide(non_zero_baseline) * 100
    )
    daily["anomaly_score"] = anomaly_score
    daily["is_anomaly"] = anomaly_score.abs() >= threshold
    daily["direction"] = "Typical"
    daily.loc[daily["is_anomaly"] & (daily["difference"] > 0), "direction"] = (
        "Spike"
    )
    daily.loc[daily["is_anomaly"] & (daily["difference"] < 0), "direction"] = (
        "Drop"
    )

    return daily.loc[
        :,
        [
            "date",
            "metric",
            "actual",
            "baseline",
            "difference",
            "percent_difference",
            "anomaly_score",
            "direction",
            "is_anomaly",
        ],
    ]


def _empty_anomaly_table() -> pd.DataFrame:
    """Return an empty result with the same columns as a populated result."""

    return pd.DataFrame(
        columns=[
            "date",
            "metric",
            "actual",
            "baseline",
            "difference",
            "percent_difference",
            "anomaly_score",
            "direction",
            "is_anomaly",
        ]
    )