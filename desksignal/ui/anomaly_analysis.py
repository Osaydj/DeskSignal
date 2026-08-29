"""Streamlit interface for deterministic anomaly analysis."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from desksignal.analytics.anomalies import detect_daily_anomalies
from desksignal.visualization.charts import daily_anomaly_figure


def _render_anomaly_metric(
    results: pd.DataFrame,
    *,
    metric: str,
) -> None:
    """Render one anomaly chart and its flagged-day table."""

    st.plotly_chart(
        daily_anomaly_figure(results, metric=metric),
        width="stretch",
        key=f"{metric}_anomaly_chart",
    )

    anomalies = results.loc[results["is_anomaly"]].copy()

    if anomalies.empty:
        st.success(f"No unusual {metric} days were found in this selection.")
        return

    st.info(
        f"Python flagged {len(anomalies):,} unusual {metric} days. "
        "A flag is a lead for investigation, not proof of a business problem."
    )

    display_table = anomalies.loc[
        :,
        [
            "date",
            "actual",
            "baseline",
            "difference",
            "percent_difference",
            "direction",
            "anomaly_score",
        ],
    ].rename(
        columns={
            "date": "Date",
            "actual": "Actual ($)",
            "baseline": "Baseline ($)",
            "difference": "Difference ($)",
            "percent_difference": "Difference (%)",
            "direction": "Direction",
            "anomaly_score": "Anomaly score",
        }
    )
    display_table["Date"] = pd.to_datetime(display_table["Date"]).dt.date
    display_table = display_table.round(
        {
            "Actual ($)": 2,
            "Baseline ($)": 2,
            "Difference ($)": 2,
            "Difference (%)": 1,
            "Anomaly score": 2,
        }
    )

    st.dataframe(
        display_table,
        width="stretch",
        hide_index=True,
    )


def render_anomaly_analysis(data: pd.DataFrame) -> None:
    """Render deterministic revenue and profit anomaly analysis."""

    st.subheader("Anomaly detection")
    st.caption(
        "Python compares each daily total with a 28-day rolling baseline. "
        "Flagged days are unusual relative to nearby dates and should be "
        "investigated with business context."
    )

    revenue_results = detect_daily_anomalies(data, metric="revenue")
    profit_results = detect_daily_anomalies(data, metric="profit")

    summary_columns = st.columns(2)
    summary_columns[0].metric(
        "Revenue anomalies",
        f"{int(revenue_results['is_anomaly'].sum()):,}",
        border=True,
    )
    summary_columns[1].metric(
        "Profit anomalies",
        f"{int(profit_results['is_anomaly'].sum()):,}",
        border=True,
    )

    revenue_tab, profit_tab = st.tabs(["Revenue", "Profit"])

    with revenue_tab:
        _render_anomaly_metric(revenue_results, metric="revenue")

    with profit_tab:
        _render_anomaly_metric(profit_results, metric="profit")
