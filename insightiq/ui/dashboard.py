"""Streamlit dashboard layout and filter controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from insightiq.analytics.kpis import BusinessKPIs, calculate_kpis
from insightiq.analytics.performance import (
    category_performance,
    daily_performance,
    filter_sales_data,
    product_performance,
    region_performance,
)
from insightiq.config import ai_features_enabled
from insightiq.ui.anomaly_analysis import render_anomaly_analysis
from insightiq.ui.ask_data import render_ask_your_data
from insightiq.ui.executive_summary import render_ai_executive_summary
from insightiq.ui.recommendations import render_ai_recommendations
from insightiq.visualization.charts import (
    profit_by_region_figure,
    profit_over_time_figure,
    revenue_by_category_figure,
    revenue_over_time_figure,
    top_products_figure,
)


@dataclass(frozen=True)
class DashboardFilters:
    """Values chosen in the Streamlit filter controls."""

    start_date: date
    end_date: date
    categories: tuple[str, ...]
    regions: tuple[str, ...]


def format_currency(value: float) -> str:
    """Format a number as US-dollar currency."""

    return f"${value:,.2f}"


def resolve_date_range(
    selected_dates: date | tuple[date, ...] | None,
    *,
    fallback_start: date,
    fallback_end: date,
) -> tuple[date, date]:
    """Return a safe range while Streamlit's range widget is being edited."""

    if selected_dates is None:
        return fallback_start, fallback_end
    if not isinstance(selected_dates, tuple):
        return selected_dates, selected_dates
    if not selected_dates:
        return fallback_start, fallback_end
    if len(selected_dates) == 1:
        return selected_dates[0], selected_dates[0]
    return min(selected_dates[0], selected_dates[1]), max(
        selected_dates[0], selected_dates[1]
    )


def render_filters(data: pd.DataFrame) -> DashboardFilters:
    """Render sidebar controls and return their selected values."""

    minimum_date = data["date"].min().date()
    maximum_date = data["date"].max().date()
    available_categories = sorted(data["category"].unique().tolist())
    available_regions = sorted(data["region"].unique().tolist())

    with st.sidebar:
        st.header("Filters")
        selected_dates = st.date_input(
            "Date range",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
        )

        start_date, end_date = resolve_date_range(
            selected_dates,
            fallback_start=minimum_date,
            fallback_end=maximum_date,
        )

        selected_categories = st.multiselect(
            "Categories",
            options=available_categories,
            help="Leave blank to include every category.",
        )
        selected_regions = st.multiselect(
            "Regions",
            options=available_regions,
            help="Leave blank to include every region.",
        )

        st.caption("Filters update every KPI and chart.")

    return DashboardFilters(
        start_date=start_date,
        end_date=end_date,
        categories=tuple(selected_categories),
        regions=tuple(selected_regions),
    )


def render_kpis(kpis: BusinessKPIs) -> None:
    """Render the eight primary business metrics."""

    first_row = st.columns(4)
    first_row[0].metric("Total revenue", format_currency(kpis.total_revenue))
    first_row[1].metric("Total profit", format_currency(kpis.total_profit))
    first_row[2].metric("Profit margin", f"{kpis.profit_margin:.1%}")
    first_row[3].metric("Transactions", f"{kpis.total_transactions:,}")

    second_row = st.columns(4)
    second_row[0].metric("Units sold", f"{kpis.total_units_sold:,}")
    second_row[1].metric(
        "Average order value", format_currency(kpis.average_order_value)
    )
    second_row[2].metric(
        "Best product",
        kpis.best_product,
        delta=f"{format_currency(kpis.best_product_revenue)} revenue",
        delta_color="off",
    )
    second_row[3].metric(
        "Best region",
        kpis.best_region,
        delta=f"{format_currency(kpis.best_region_revenue)} revenue",
        delta_color="off",
    )


def render_accessible_analytics(data: pd.DataFrame) -> None:
    """Provide chart-equivalent tables and a filtered-data download."""

    with st.expander("View analytics tables and downloads"):
        st.caption(
            "These tables provide the same grouped results as the charts in a "
            "screen-reader-friendly format."
        )
        daily_tab, category_tab, region_tab, product_tab = st.tabs(
            ["Daily trends", "Categories", "Regions", "Products"]
        )
        with daily_tab:
            st.dataframe(daily_performance(data), width="stretch", hide_index=True)
        with category_tab:
            st.dataframe(category_performance(data), width="stretch", hide_index=True)
        with region_tab:
            st.dataframe(region_performance(data), width="stretch", hide_index=True)
        with product_tab:
            st.dataframe(
                product_performance(data, top_n=data["product"].nunique()),
                width="stretch",
                hide_index=True,
            )

        st.download_button(
            "Download filtered CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="insightiq_filtered_sales.csv",
            mime="text/csv",
        )


def render_optional_ai(data: pd.DataFrame) -> None:
    """Render Claude features only when an operator explicitly enables them."""

    if ai_features_enabled():
        render_ai_executive_summary(data)
        render_ai_recommendations(data)
        render_ask_your_data(data)
        return

    st.divider()
    st.header("Optional AI features")
    st.info(
        "AI Executive Summary, AI Recommendations, and Ask Your Data are disabled "
        "in analytics-only mode. The dashboard and every displayed statistic are "
        "still fully calculated by Python."
    )

def render_dashboard(data: pd.DataFrame) -> None:
    """Render the filtered analytics dashboard in organized tabs."""

    filters = render_filters(data)
    filtered_data = filter_sales_data(
        data,
        start_date=filters.start_date,
        end_date=filters.end_date,
        categories=filters.categories,
        regions=filters.regions,
    )

    if filtered_data.empty:
        st.warning("No transactions match the selected filters.")
        return

    st.caption(
        f"Showing {len(filtered_data):,} rows and "
        f"{filtered_data['transaction_id'].nunique():,} distinct transactions."
    )

    (
        overview_tab,
        trends_tab,
        anomalies_tab,
        performance_tab,
        data_tab,
        ai_tab,
    ) = st.tabs(
        [
            "Overview",
            "Trends",
            "Anomalies",
            "Products & Markets",
            "Data Explorer",
            "AI Insights",
        ]
    )

    with overview_tab:
        st.subheader("Business overview")
        st.caption("A summary of the most important results for the selected data.")
        render_kpis(calculate_kpis(filtered_data))

    with trends_tab:
        st.subheader("Revenue and profit trends")
        st.caption("Explore how revenue and profit changed over the selected period.")

        trend_columns = st.columns(2)

        with trend_columns[0]:
            st.plotly_chart(
                revenue_over_time_figure(filtered_data),
                width="stretch",
                key="revenue_over_time",
            )

        with trend_columns[1]:
            st.plotly_chart(
                profit_over_time_figure(filtered_data),
                width="stretch",
                key="profit_over_time",
            )
    with anomalies_tab:
            render_anomaly_analysis(filtered_data)

    with performance_tab:
        st.subheader("Products and markets")
        st.caption(
            "Compare category revenue, regional profit, and product performance."
        )

        performance_columns = st.columns(2)

        with performance_columns[0]:
            st.plotly_chart(
                revenue_by_category_figure(filtered_data),
                width="stretch",
                key="revenue_by_category",
            )

        with performance_columns[1]:
            st.plotly_chart(
                profit_by_region_figure(filtered_data),
                width="stretch",
                key="profit_by_region",
            )

        st.plotly_chart(
            top_products_figure(filtered_data),
            width="stretch",
            key="top_products",
        )

    with data_tab:
        st.subheader("Data explorer")
        st.caption(
            "Review accessible analytics tables and inspect the filtered transactions."
        )

        render_accessible_analytics(filtered_data)

        with st.expander("View filtered data"):
            st.dataframe(
                filtered_data,
                width="stretch",
                hide_index=True,
            )

    with ai_tab:
        render_optional_ai(filtered_data)
