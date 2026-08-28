"""Tests for deterministic KPI and grouped calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from insightiq.analytics.kpis import calculate_kpis
from insightiq.analytics.executive import (
    monthly_performance,
    performance_by_dimension,
    unusual_monthly_changes,
)
from insightiq.analytics.performance import (
    category_performance,
    daily_performance,
    filter_sales_data,
    product_performance,
    region_performance,
)
from insightiq.data.validation import validate_sales_data


def test_kpis_match_known_answers(valid_sales_data: pd.DataFrame) -> None:
    data = validate_sales_data(valid_sales_data)
    kpis = calculate_kpis(data)

    assert kpis.total_revenue == pytest.approx(80.0)
    assert kpis.total_profit == pytest.approx(30.0)
    assert kpis.profit_margin == pytest.approx(0.375)
    assert kpis.total_transactions == 2
    assert kpis.total_units_sold == 6
    assert kpis.average_order_value == pytest.approx(40.0)
    assert kpis.best_product == "Alpha"
    assert kpis.best_product_revenue == pytest.approx(50.0)
    assert kpis.best_region == "West"
    assert kpis.best_region_revenue == pytest.approx(60.0)


def test_filters_are_inclusive(valid_sales_data: pd.DataFrame) -> None:
    data = validate_sales_data(valid_sales_data)

    filtered = filter_sales_data(
        data,
        start_date="2025-01-02",
        end_date="2025-01-02",
        categories=["Accessories"],
        regions=["West"],
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["product"] == "Alpha"


def test_grouped_summaries_reconcile_to_totals(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    assert daily_performance(data)["revenue"].sum() == pytest.approx(80.0)
    assert category_performance(data)["revenue"].sum() == pytest.approx(80.0)
    assert region_performance(data)["profit"].sum() == pytest.approx(30.0)
    assert product_performance(data)["revenue"].sum() == pytest.approx(80.0)


def test_invalid_date_range_is_rejected(valid_sales_data: pd.DataFrame) -> None:
    data = validate_sales_data(valid_sales_data)

    with pytest.raises(ValueError, match="start_date"):
        filter_sales_data(
            data,
            start_date="2025-02-01",
            end_date="2025-01-01",
        )


def test_monthly_performance_calculates_changes() -> None:
    data = pd.DataFrame(
        {
            "transaction_id": ["T-1", "T-2", "T-3"],
            "date": pd.to_datetime(["2025-01-10", "2025-02-10", "2025-03-10"]),
            "revenue": [100.0, 125.0, 100.0],
            "profit": [40.0, 50.0, 20.0],
            "units_sold": [2, 3, 2],
        }
    )

    monthly = monthly_performance(data)

    assert monthly.loc[1, "revenue_change_pct"] == pytest.approx(25.0)
    assert monthly.loc[2, "revenue_change_pct"] == pytest.approx(-20.0)
    assert monthly.loc[2, "profit_margin_pct"] == pytest.approx(20.0)

    unusual = unusual_monthly_changes(monthly, threshold_pct=20.0)
    assert set(unusual["metric"]) == {"revenue", "profit"}
    assert len(unusual) == 4


def test_dimension_performance_includes_margin_and_share(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    categories = performance_by_dimension(data, "category")

    assert categories["revenue_share_pct"].sum() == pytest.approx(100.0)
    accessories = categories.loc[
        categories["category"] == "Accessories"
    ].iloc[0]
    assert accessories["profit_margin_pct"] == pytest.approx(40.0)


def test_unsupported_performance_dimension_is_rejected(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    with pytest.raises(ValueError, match="dimension"):
        performance_by_dimension(data, "customer_type")
