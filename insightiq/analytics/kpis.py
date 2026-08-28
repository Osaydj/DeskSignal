"""Reusable KPI calculations for InsightIQ."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BusinessKPIs:
    """Calculated business metrics for one filtered dataset."""

    total_revenue: float
    total_profit: float
    profit_margin: float
    total_transactions: int
    total_units_sold: int
    average_order_value: float
    best_product: str
    best_product_revenue: float
    best_region: str
    best_region_revenue: float


def _best_group_by_revenue(data: pd.DataFrame, column: str) -> tuple[str, float]:
    """Return the highest-revenue label and value with stable tie-breaking."""

    grouped = (
        data.groupby(column, as_index=False, observed=True)["revenue"]
        .sum()
        .sort_values(["revenue", column], ascending=[False, True], ignore_index=True)
    )
    winner = grouped.iloc[0]
    return str(winner[column]), float(winner["revenue"])


def calculate_kpis(data: pd.DataFrame) -> BusinessKPIs:
    """Calculate top-level KPIs from validated, filtered sales data.

    Best product and best region are defined by total revenue. Transactions are
    counted with distinct transaction IDs, so line-item datasets also work.
    """

    if data.empty:
        raise ValueError("KPI calculations require at least one sales row.")

    total_revenue = float(data["revenue"].sum())
    total_profit = float(data["profit"].sum())
    total_transactions = int(data["transaction_id"].nunique())
    total_units_sold = int(data["units_sold"].sum())

    profit_margin = total_profit / total_revenue if total_revenue else 0.0
    average_order_value = (
        total_revenue / total_transactions if total_transactions else 0.0
    )
    best_product, best_product_revenue = _best_group_by_revenue(data, "product")
    best_region, best_region_revenue = _best_group_by_revenue(data, "region")

    return BusinessKPIs(
        total_revenue=total_revenue,
        total_profit=total_profit,
        profit_margin=profit_margin,
        total_transactions=total_transactions,
        total_units_sold=total_units_sold,
        average_order_value=average_order_value,
        best_product=best_product,
        best_product_revenue=best_product_revenue,
        best_region=best_region,
        best_region_revenue=best_region_revenue,
    )
