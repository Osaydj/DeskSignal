"""Filtering and grouped performance calculations."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import pandas as pd


DateLike = date | datetime | pd.Timestamp


def filter_sales_data(
    data: pd.DataFrame,
    *,
    start_date: DateLike | None = None,
    end_date: DateLike | None = None,
    categories: Sequence[str] | None = None,
    regions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return sales rows matching the selected inclusive filters."""

    if start_date is not None and end_date is not None:
        if pd.Timestamp(start_date) > pd.Timestamp(end_date):
            raise ValueError("start_date cannot be after end_date.")

    mask = pd.Series(True, index=data.index)

    if start_date is not None:
        mask &= data["date"] >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= data["date"].dt.normalize() <= pd.Timestamp(end_date).normalize()
    if categories:
        mask &= data["category"].isin(categories)
    if regions:
        mask &= data["region"].isin(regions)

    return data.loc[mask].copy().reset_index(drop=True)


def daily_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue and profit by calendar date."""

    daily = data.copy()
    daily["date"] = daily["date"].dt.normalize()
    return (
        daily.groupby("date", as_index=False, observed=True)
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
        .sort_values("date", ignore_index=True)
    )


def category_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue and profit by product category."""

    return (
        data.groupby("category", as_index=False, observed=True)
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
        .sort_values(["revenue", "category"], ascending=[False, True], ignore_index=True)
    )


def region_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue and profit by region."""

    return (
        data.groupby("region", as_index=False, observed=True)
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
        .sort_values(["profit", "region"], ascending=[False, True], ignore_index=True)
    )


def product_performance(data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top products ranked by revenue."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    return (
        data.groupby("product", as_index=False, observed=True)
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            units_sold=("units_sold", "sum"),
        )
        .sort_values(["revenue", "product"], ascending=[False, True], ignore_index=True)
        .head(top_n)
        .reset_index(drop=True)
    )
