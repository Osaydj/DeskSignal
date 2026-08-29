"""Streamlit dashboard layout and filter controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from desksignal.analytics.kpis import BusinessKPIs, calculate_kpis
from desksignal.analytics.performance import (
    category_performance,
    daily_performance,
    filter_sales_data,
    product_performance,
    region_performance,
)
from desksignal.config import ai_features_enabled
from desksignal.ui.anomaly_analysis import render_anomaly_analysis
from desksignal.ui.ai_insights import render_local_ai_insights
from desksignal.ui.ask_data import render_ask_your_data
from desksignal.ui.downloads import dataframe_to_safe_csv
from desksignal.ui.executive_summary import render_ai_executive_summary
from desksignal.ui.forecasting import render_forecasting
from desksignal.ui.formatting import format_currency
from desksignal.ui.recommendations import render_ai_recommendations
from desksignal.visualization.charts import (
    profit_by_region_figure,
    profit_over_time_figure,
    revenue_by_category_figure,
    revenue_over_time_figure,
    top_products_figure,
)


FILTER_DATE_KEY = "dashboard_date_range"
FILTER_CATEGORY_KEY = "dashboard_categories"
FILTER_REGION_KEY = "dashboard_regions"
FILTER_SOURCE_KEY = "_dashboard_filter_source"


@dataclass(frozen=True)
class DashboardFilters:
    """Values chosen in the Streamlit filter controls."""

    start_date: date
    end_date: date
    categories: tuple[str, ...]
    regions: tuple[str, ...]


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


def _reset_filter_widgets(minimum_date: date, maximum_date: date) -> None:
    """Restore dashboard filter widgets to the full dataset scope."""

    st.session_state[FILTER_DATE_KEY] = (minimum_date, maximum_date)
    st.session_state[FILTER_CATEGORY_KEY] = []
    st.session_state[FILTER_REGION_KEY] = []


def render_filters(data: pd.DataFrame) -> DashboardFilters:
    """Render sidebar controls and return their selected values."""

    minimum_date = data["date"].min().date()
    maximum_date = data["date"].max().date()
    available_categories = sorted(data["category"].unique().tolist())
    available_regions = sorted(data["region"].unique().tolist())
    source_signature = (
        minimum_date,
        maximum_date,
        tuple(available_categories),
        tuple(available_regions),
    )

    if st.session_state.get(FILTER_SOURCE_KEY) != source_signature:
        st.session_state[FILTER_SOURCE_KEY] = source_signature
        _reset_filter_widgets(minimum_date, maximum_date)

    with st.sidebar:
        st.subheader("Dataset")
        st.badge(
            "Validated sales data",
            icon=":material/verified:",
            color="green",
        )
        st.caption(
            f"{len(data):,} rows · "
            f"{data['transaction_id'].nunique():,} transactions"
        )
        st.caption(
            f"{minimum_date:%b %d, %Y} — {maximum_date:%b %d, %Y}"
        )
        st.divider()
        st.header("Filters")
        selected_dates = st.date_input(
            "Date range",
            min_value=minimum_date,
            max_value=maximum_date,
            key=FILTER_DATE_KEY,
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
            key=FILTER_CATEGORY_KEY,
        )
        selected_regions = st.multiselect(
            "Regions",
            options=available_regions,
            help="Leave blank to include every region.",
            key=FILTER_REGION_KEY,
        )

        st.button(
            "Reset filters",
            icon=":material/restart_alt:",
            width="stretch",
            on_click=_reset_filter_widgets,
            args=(minimum_date, maximum_date),
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
    first_row[0].metric(
        "Total revenue",
        format_currency(kpis.total_revenue),
        border=True,
    )
    first_row[1].metric(
        "Total profit",
        format_currency(kpis.total_profit),
        border=True,
    )
    first_row[2].metric(
        "Profit margin",
        f"{kpis.profit_margin:.1%}",
        border=True,
    )
    first_row[3].metric(
        "Transactions",
        f"{kpis.total_transactions:,}",
        border=True,
    )

    second_row = st.columns(4)
    second_row[0].metric(
        "Units sold",
        f"{kpis.total_units_sold:,}",
        border=True,
    )
    second_row[1].metric(
        "Average order value",
        format_currency(kpis.average_order_value),
        border=True,
    )
    second_row[2].metric(
        "Best product",
        kpis.best_product,
        delta=f"{format_currency(kpis.best_product_revenue)} revenue",
        delta_color="off",
        border=True,
    )
    second_row[3].metric(
        "Best region",
        kpis.best_region,
        delta=f"{format_currency(kpis.best_region_revenue)} revenue",
        delta_color="off",
        border=True,
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
            data=dataframe_to_safe_csv(data),
            file_name="desksignal_filtered_sales.csv",
            mime="text/csv",
            icon=":material/download:",
        )


def render_optional_ai(data: pd.DataFrame) -> None:
    """Render Claude features only when an operator explicitly enables them."""

    if ai_features_enabled():
        render_ai_executive_summary(data)
        render_ai_recommendations(data)
        render_ask_your_data(data)
        return

    st.divider()
    st.subheader("Optional Claude tools")
    st.info(
        "Claude Executive Summary, Recommendations, and Ask Your Data are not "
        "configured. The local machine-learning tools above remain fully available "
        "without an API key."
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
        forecasting_tab,
        performance_tab,
        data_tab,
        ai_tab,
    ) = st.tabs(
        [
            "Overview",
            "Trends",
            "Anomalies",
            "Forecasting",
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

    with forecasting_tab:
        render_forecasting(filtered_data)

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
        render_local_ai_insights(filtered_data)
        render_optional_ai(filtered_data)
