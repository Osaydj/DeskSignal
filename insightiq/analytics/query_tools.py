"""Read-only Pandas calculations exposed to the Ask Your Data tool layer."""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd


AdditiveMetric = Literal["revenue", "profit", "units_sold", "transactions"]
RankingMetric = Literal[
    "revenue",
    "profit",
    "profit_margin",
    "units_sold",
    "transactions",
]
Dimension = Literal["product", "region", "category"]
Frequency = Literal["day", "week", "month"]


def apply_analysis_filters(
    data: pd.DataFrame,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    product: str = "",
    category: str = "",
    region: str = "",
) -> pd.DataFrame:
    """Return rows matching safe, exact-value filters.

    Filters use normal Pandas comparisons. Uploaded values are never treated as
    Python expressions, SQL, regular expressions, or executable code.
    """

    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date cannot be after end_date.")

    mask = pd.Series(True, index=data.index)
    if start_date is not None:
        mask &= data["date"] >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= data["date"].dt.normalize() <= pd.Timestamp(end_date)
    if product:
        mask &= data["product"].eq(product)
    if category:
        mask &= data["category"].eq(category)
    if region:
        mask &= data["region"].eq(region)

    return data.loc[mask].copy().reset_index(drop=True)


def calculate_additive_metric(data: pd.DataFrame, metric: AdditiveMetric) -> float | int:
    """Calculate one supported additive business metric."""

    if metric == "transactions":
        return int(data["transaction_id"].nunique())
    if metric == "units_sold":
        return int(data["units_sold"].sum())
    return float(data[metric].sum())


def calculate_profit_margin(data: pd.DataFrame) -> float:
    """Return profit as a percentage of revenue."""

    revenue = float(data["revenue"].sum())
    return float(data["profit"].sum()) / revenue * 100 if revenue else 0.0


def _percent_change(first_value: float, second_value: float) -> float | None:
    """Return change relative to the first value, or None for a zero baseline."""

    if first_value == 0:
        return None
    return (second_value - first_value) / abs(first_value) * 100


def compare_periods(
    data: pd.DataFrame,
    *,
    period_1_start: date,
    period_1_end: date,
    period_2_start: date,
    period_2_end: date,
    metric: AdditiveMetric,
    group_by: Literal["overall", "product", "region", "category"],
    limit: int,
) -> dict[str, object]:
    """Compare two inclusive periods overall or by a business dimension."""

    first = apply_analysis_filters(
        data,
        start_date=period_1_start,
        end_date=period_1_end,
    )
    second = apply_analysis_filters(
        data,
        start_date=period_2_start,
        end_date=period_2_end,
    )
    first_value = float(calculate_additive_metric(first, metric))
    second_value = float(calculate_additive_metric(second, metric))
    overall = {
        "period_1_value": first_value,
        "period_2_value": second_value,
        "absolute_change": second_value - first_value,
        "percent_change": _percent_change(first_value, second_value),
    }

    if group_by == "overall":
        return {"overall": overall, "largest_increases": [], "largest_declines": []}

    def grouped_values(period_data: pd.DataFrame) -> pd.DataFrame:
        if metric == "transactions":
            grouped = period_data.groupby(group_by, observed=True)[
                "transaction_id"
            ].nunique()
        else:
            grouped = period_data.groupby(group_by, observed=True)[metric].sum()
        return grouped.rename("value").reset_index()

    first_grouped = grouped_values(first).rename(columns={"value": "period_1_value"})
    second_grouped = grouped_values(second).rename(columns={"value": "period_2_value"})
    comparison = first_grouped.merge(second_grouped, on=group_by, how="outer").fillna(0)
    comparison["absolute_change"] = (
        comparison["period_2_value"] - comparison["period_1_value"]
    )
    comparison["percent_change"] = comparison.apply(
        lambda row: _percent_change(
            float(row["period_1_value"]),
            float(row["period_2_value"]),
        ),
        axis=1,
    )

    increases = comparison.loc[comparison["absolute_change"] > 0].sort_values(
        ["absolute_change", group_by],
        ascending=[False, True],
    )
    declines = comparison.loc[comparison["absolute_change"] < 0].sort_values(
        ["absolute_change", group_by],
        ascending=[True, True],
    )

    return {
        "overall": overall,
        "largest_increases": increases.head(limit).to_dict(orient="records"),
        "largest_declines": declines.head(limit).to_dict(orient="records"),
    }


def dimension_performance(
    data: pd.DataFrame,
    *,
    dimension: Dimension,
    metric: RankingMetric,
    order: Literal["highest", "lowest"],
    limit: int,
) -> pd.DataFrame:
    """Rank products, regions, or categories using deterministic calculations."""

    grouped = (
        data.groupby(dimension, as_index=False, observed=True)
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            units_sold=("units_sold", "sum"),
            transactions=("transaction_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    grouped["profit_margin"] = np.where(
        grouped["revenue"].ne(0),
        grouped["profit"] / grouped["revenue"] * 100,
        0.0,
    )
    ascending = order == "lowest"
    return (
        grouped.sort_values(
            [metric, dimension],
            ascending=[ascending, True],
            ignore_index=True,
        )
        .head(limit)
        .reset_index(drop=True)
    )


def get_time_series(
    data: pd.DataFrame,
    *,
    metric: AdditiveMetric,
    frequency: Frequency,
) -> pd.DataFrame:
    """Aggregate one metric by day, week, or month."""

    series = data.copy()
    if frequency == "day":
        series["period"] = series["date"].dt.normalize()
    elif frequency == "week":
        series["period"] = series["date"].dt.to_period("W-SUN").dt.start_time
    else:
        series["period"] = series["date"].dt.to_period("M").dt.to_timestamp()

    if metric == "transactions":
        grouped = series.groupby("period", as_index=False, observed=True).agg(
            value=("transaction_id", "nunique")
        )
    else:
        grouped = series.groupby("period", as_index=False, observed=True).agg(
            value=(metric, "sum")
        )
    return grouped.sort_values("period", ignore_index=True)
