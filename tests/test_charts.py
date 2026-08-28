"""Tests for Plotly chart builders."""

from __future__ import annotations

import pandas as pd

from insightiq.data.validation import validate_sales_data
from insightiq.visualization.charts import (
    profit_by_region_figure,
    profit_over_time_figure,
    revenue_by_category_figure,
    revenue_over_time_figure,
    top_products_figure,
)


def test_every_requested_chart_contains_data(valid_sales_data: pd.DataFrame) -> None:
    data = validate_sales_data(valid_sales_data)
    figures = [
        revenue_over_time_figure(data),
        profit_over_time_figure(data),
        revenue_by_category_figure(data),
        profit_by_region_figure(data),
        top_products_figure(data),
    ]

    assert all(len(figure.data) >= 1 for figure in figures)
    assert all(figure.layout.title.text for figure in figures)


def test_top_product_names_do_not_receive_currency_prefix(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    figure = top_products_figure(data)

    assert figure.layout.yaxis.tickprefix in (None, "")
    assert figure.layout.xaxis.tickprefix == "$"
