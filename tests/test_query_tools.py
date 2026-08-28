"""Tests for deterministic calculations exposed through Ask Your Data."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from insightiq.analytics.query_tools import (
    apply_analysis_filters,
    compare_periods,
    dimension_performance,
    get_time_series,
)
from insightiq.data.validation import validate_sales_data


def test_exact_filters_never_evaluate_uploaded_text(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    filtered = apply_analysis_filters(data, region="West")
    suspicious = apply_analysis_filters(data, region="__import__('os').system('x')")

    assert filtered["revenue"].sum() == pytest.approx(60.0)
    assert suspicious.empty


def test_dimension_performance_can_find_highest_and_lowest(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    strongest_region = dimension_performance(
        data,
        dimension="region",
        metric="profit",
        order="highest",
        limit=1,
    )
    lowest_margin_category = dimension_performance(
        data,
        dimension="category",
        metric="profit_margin",
        order="lowest",
        limit=1,
    )

    assert strongest_region.iloc[0]["region"] == "West"
    assert strongest_region.iloc[0]["profit"] == pytest.approx(22.0)
    assert lowest_margin_category.iloc[0]["category"] == "Electronics"
    assert lowest_margin_category.iloc[0]["profit_margin"] == pytest.approx(
        100 / 3
    )


def test_period_comparison_reports_change_and_contributors(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    comparison = compare_periods(
        data,
        period_1_start=date(2025, 1, 1),
        period_1_end=date(2025, 1, 1),
        period_2_start=date(2025, 1, 2),
        period_2_end=date(2025, 1, 2),
        metric="revenue",
        group_by="product",
        limit=5,
    )

    assert comparison["overall"]["period_1_value"] == pytest.approx(20.0)
    assert comparison["overall"]["period_2_value"] == pytest.approx(60.0)
    assert comparison["overall"]["percent_change"] == pytest.approx(200.0)
    assert comparison["largest_increases"][0]["product"] == "Beta"


def test_time_series_uses_requested_frequency(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    daily = get_time_series(data, metric="profit", frequency="day")
    monthly = get_time_series(data, metric="profit", frequency="month")

    assert daily["value"].tolist() == pytest.approx([8.0, 22.0])
    assert len(monthly) == 1
    assert monthly.iloc[0]["value"] == pytest.approx(30.0)
