"""Deterministic calculations used by the AI executive summary."""

from __future__ import annotations

import numpy as np
import pandas as pd


SUPPORTED_DIMENSIONS = frozenset({"category", "product", "region"})


def _safe_percent_change(values: pd.Series) -> pd.Series:
    """Calculate percent changes without returning infinity for zero baselines."""

    previous = values.shift(1)
    changes = (values - previous) / previous.abs() * 100
    return changes.replace([np.inf, -np.inf], np.nan)


def monthly_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate business performance by calendar month.

    Change columns compare each month with the immediately previous month. A
    change is left blank when there is no previous month or the prior value is
    zero, because a meaningful percentage cannot be calculated in those cases.
    """

    monthly = data.assign(
        month=data["date"].dt.to_period("M").dt.to_timestamp()
    )
    monthly = (
        monthly.groupby("month", as_index=False, observed=True)
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            transactions=("transaction_id", "nunique"),
            units_sold=("units_sold", "sum"),
        )
        .sort_values("month", ignore_index=True)
    )

    monthly["profit_margin_pct"] = np.where(
        monthly["revenue"].ne(0),
        monthly["profit"] / monthly["revenue"] * 100,
        0.0,
    )
    monthly["revenue_change_pct"] = _safe_percent_change(monthly["revenue"])
    monthly["profit_change_pct"] = _safe_percent_change(monthly["profit"])
    return monthly


def performance_by_dimension(
    data: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    """Aggregate revenue, profit, units, and transactions for one dimension."""

    if dimension not in SUPPORTED_DIMENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DIMENSIONS))
        raise ValueError(f"dimension must be one of: {supported}.")

    grouped = (
        data.groupby(dimension, as_index=False, observed=True)
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            units_sold=("units_sold", "sum"),
            transactions=("transaction_id", "nunique"),
        )
        .sort_values(
            ["revenue", dimension],
            ascending=[False, True],
            ignore_index=True,
        )
    )

    total_revenue = float(grouped["revenue"].sum())
    grouped["revenue_share_pct"] = (
        grouped["revenue"] / total_revenue * 100 if total_revenue else 0.0
    )
    grouped["profit_margin_pct"] = np.where(
        grouped["revenue"].ne(0),
        grouped["profit"] / grouped["revenue"] * 100,
        0.0,
    )
    return grouped


def unusual_monthly_changes(
    monthly: pd.DataFrame,
    *,
    threshold_pct: float = 20.0,
) -> pd.DataFrame:
    """Flag large month-over-month revenue or profit changes.

    This is a transparent rule-based flag, not a statistical anomaly detector.
    A row is returned when the absolute percentage change meets the threshold.
    """

    if threshold_pct <= 0:
        raise ValueError("threshold_pct must be greater than zero.")

    records: list[dict[str, object]] = []
    metric_columns = {
        "revenue": "revenue_change_pct",
        "profit": "profit_change_pct",
    }

    for index in range(1, len(monthly)):
        previous = monthly.iloc[index - 1]
        current = monthly.iloc[index]

        for metric, change_column in metric_columns.items():
            change = current[change_column]
            if pd.isna(change) or abs(float(change)) < threshold_pct:
                continue

            records.append(
                {
                    "metric": metric,
                    "from_month": previous["month"],
                    "to_month": current["month"],
                    "previous_value": float(previous[metric]),
                    "current_value": float(current[metric]),
                    "change_pct": float(change),
                }
            )

    return pd.DataFrame(
        records,
        columns=[
            "metric",
            "from_month",
            "to_month",
            "previous_value",
            "current_value",
            "change_pct",
        ],
    )
