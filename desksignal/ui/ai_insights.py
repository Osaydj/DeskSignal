"""Streamlit interface for no-API, locally trained machine-learning tools."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from desksignal.analytics.machine_learning import (
    MLForecast,
    MarketSegmentation,
    cluster_market_segments,
    create_ml_forecast,
    detect_ml_daily_anomalies,
)
from desksignal.ui.downloads import dataframe_to_safe_csv
from desksignal.ui.formatting import format_currency
from desksignal.visualization.machine_learning import (
    market_segments_figure,
    ml_anomaly_figure,
    ml_forecast_figure,
)


ANOMALY_SENSITIVITIES = {
    "Focused (2% flagged)": 0.02,
    "Balanced (4% flagged)": 0.04,
    "Broad (8% flagged)": 0.08,
}
ML_FORECAST_HORIZONS = (7, 14, 30)


@st.cache_data(show_spinner=False, max_entries=12)
def _cached_anomalies(
    data: pd.DataFrame, contamination: float
) -> pd.DataFrame:
    """Cache identical local anomaly-model runs."""

    return detect_ml_daily_anomalies(data, contamination=contamination)


@st.cache_data(show_spinner=False, max_entries=12)
def _cached_segments(data: pd.DataFrame, n_clusters: int) -> MarketSegmentation:
    """Cache identical local segmentation runs."""

    return cluster_market_segments(data, n_clusters=n_clusters)


@st.cache_data(show_spinner=False, max_entries=12)
def _cached_forecast(
    data: pd.DataFrame, metric: str, horizon_days: int
) -> MLForecast:
    """Cache identical local forecast-model runs."""

    return create_ml_forecast(
        data,
        metric=metric,
        horizon_days=horizon_days,
    )


def _render_ml_anomalies(data: pd.DataFrame) -> None:
    """Render Isolation Forest controls, evidence, and downloads."""

    sensitivity_label = st.selectbox(
        "Detection sensitivity",
        options=list(ANOMALY_SENSITIVITIES),
        index=1,
        help="A broader setting flags more days for investigation.",
        key="ml_anomaly_sensitivity",
    )
    metric = st.radio(
        "Chart metric",
        options=("revenue", "profit"),
        horizontal=True,
        format_func=str.title,
        key="ml_anomaly_metric",
    )

    try:
        results = _cached_anomalies(
            data,
            ANOMALY_SENSITIVITIES[sensitivity_label],
        )
    except ValueError as error:
        st.warning(str(error))
        return

    flagged = results.loc[results["is_anomaly"]].copy()
    metrics = st.columns(3)
    metrics[0].metric("Model", "Isolation Forest", border=True)
    metrics[1].metric("Flagged days", f"{len(flagged):,}", border=True)
    metrics[2].metric(
        "Share of days",
        f"{len(flagged) / len(results):.1%}",
        border=True,
    )
    st.plotly_chart(
        ml_anomaly_figure(results, metric=metric),
        width="stretch",
        key="ml_anomaly_chart",
    )
    st.info(
        "The model evaluates revenue, profit, units, transactions, order value, "
        "and margin together. A flag is an investigation lead—not proof of an error."
    )

    display = flagged.loc[
        :,
        [
            "date",
            "revenue",
            "profit",
            "units_sold",
            "transactions",
            "anomaly_score",
            "direction",
        ],
    ].rename(
        columns={
            "date": "Date",
            "revenue": "Revenue ($)",
            "profit": "Profit ($)",
            "units_sold": "Units",
            "transactions": "Transactions",
            "anomaly_score": "Anomaly score",
            "direction": "Revenue direction",
        }
    )
    display["Date"] = pd.to_datetime(display["Date"]).dt.date
    display = display.round(2)
    with st.expander("Review ML-flagged days"):
        st.dataframe(display, width="stretch", hide_index=True)
        st.download_button(
            "Download ML anomaly results",
            data=dataframe_to_safe_csv(results),
            file_name="desksignal_ml_anomalies.csv",
            mime="text/csv",
            icon=":material/download:",
            key="download_ml_anomalies",
        )


def _render_market_segments(data: pd.DataFrame) -> None:
    """Render K-Means controls, segment profiles, and downloads."""

    n_clusters = int(
        st.selectbox(
            "Number of market clusters",
            options=(2, 3, 4, 5, 6),
            index=2,
            help="Clusters group similar product, category, region, and customer combinations.",
            key="ml_cluster_count",
        )
    )
    try:
        result = _cached_segments(data, n_clusters)
    except ValueError as error:
        st.warning(str(error))
        return

    metrics = st.columns(3)
    metrics[0].metric("Model", "K-Means", border=True)
    metrics[1].metric(
        "Product-market segments",
        f"{len(result.assignments):,}",
        border=True,
    )
    metrics[2].metric(
        "Silhouette score",
        f"{result.silhouette_score:.2f}",
        help="Higher values indicate more clearly separated groups; context still matters.",
        border=True,
    )
    st.plotly_chart(
        market_segments_figure(result.assignments),
        width="stretch",
        key="ml_segment_chart",
    )
    st.caption(
        "Bubble size represents transaction volume. Cluster numbers are ordered by "
        "total revenue so the results stay easy to compare."
    )

    summary = result.summary.rename(
        columns={
            "cluster": "Cluster",
            "profile": "Profile",
            "segment_count": "Segments",
            "revenue": "Revenue ($)",
            "profit": "Profit ($)",
            "profit_margin": "Profit margin",
            "transactions": "Transactions",
            "units_sold": "Units",
        }
    ).copy()
    summary["Profit margin"] = summary["Profit margin"].map(
        lambda value: f"{value:.1%}"
    )
    summary = summary.round(2)
    st.dataframe(summary, width="stretch", hide_index=True)

    with st.expander("Review every segment assignment"):
        st.dataframe(result.assignments.round(2), width="stretch", hide_index=True)
        st.download_button(
            "Download market segment assignments",
            data=dataframe_to_safe_csv(result.assignments),
            file_name="desksignal_market_segments.csv",
            mime="text/csv",
            icon=":material/download:",
            key="download_ml_segments",
        )


def _render_ml_forecast(data: pd.DataFrame) -> None:
    """Render Random Forest controls, validation evidence, and downloads."""

    controls = st.columns(2)
    with controls[0]:
        metric = st.selectbox(
            "Forecast metric",
            options=("revenue", "profit"),
            format_func=str.title,
            key="ml_forecast_metric",
        )
    with controls[1]:
        horizon_days = int(
            st.selectbox(
                "ML forecast horizon",
                options=ML_FORECAST_HORIZONS,
                index=1,
                format_func=lambda days: f"{days} days",
                key="ml_forecast_horizon",
            )
        )

    try:
        result = _cached_forecast(data, metric, horizon_days)
    except ValueError as error:
        st.warning(str(error))
        return

    forecast_total = float(result.forecast["forecast"].sum())
    metrics = st.columns(4)
    metrics[0].metric(
        f"{horizon_days}-day forecast",
        format_currency(forecast_total),
        border=True,
    )
    metrics[1].metric(
        "Validation MAE",
        format_currency(result.mean_absolute_error),
        help="Mean absolute error on the newest held-out historical days.",
        border=True,
    )
    metrics[2].metric(
        "Validation WAPE",
        f"{result.weighted_absolute_percentage_error:.1f}%",
        border=True,
    )
    metrics[3].metric(
        "vs. 7-day baseline",
        f"{result.improvement_over_baseline:+.1f}%",
        help="Positive means lower validation error than repeating the value from seven days earlier.",
        border=True,
    )
    st.plotly_chart(
        ml_forecast_figure(result),
        width="stretch",
        key="ml_forecast_chart",
    )
    st.info(
        "This model is validated on the newest historical days before forecasting. "
        "Results are estimates for decision support, not guaranteed outcomes."
    )

    feature_names = {
        "trend_index": "Long-term trend",
        "day_of_week": "Day of week",
        "month": "Month",
        "day_of_year_sin": "Annual seasonality (sin)",
        "day_of_year_cos": "Annual seasonality (cos)",
        "lag_1": "Previous day",
        "lag_7": "Previous week",
        "lag_14": "Previous 2 weeks",
        "lag_28": "Previous 4 weeks",
        "rolling_mean_7": "7-day average",
        "rolling_mean_28": "28-day average",
        "rolling_std_7": "7-day variability",
    }
    importance = result.feature_importance.head(8).copy()
    importance["feature"] = importance["feature"].map(feature_names)
    importance = importance.set_index("feature")["importance"]

    with st.expander("Explain the model and view forecast values"):
        st.markdown("**Most influential model inputs**")
        st.bar_chart(importance, horizontal=True)
        st.caption(
            "Feature importance shows what the model used most; it does not prove cause and effect."
        )
        display = result.forecast.rename(
            columns={
                "date": "Date",
                "forecast": "Forecast ($)",
                "lower_bound": "Lower estimate ($)",
                "upper_bound": "Upper estimate ($)",
            }
        ).copy()
        display["Date"] = pd.to_datetime(display["Date"]).dt.date
        st.dataframe(display.round(2), width="stretch", hide_index=True)
        st.download_button(
            "Download ML forecast",
            data=dataframe_to_safe_csv(result.forecast),
            file_name=f"desksignal_{metric}_{horizon_days}_day_ml_forecast.csv",
            mime="text/csv",
            icon=":material/download:",
            key="download_ml_forecast",
        )


def render_local_ai_insights(data: pd.DataFrame) -> None:
    """Render DeskSignal's three local machine-learning tools."""

    st.subheader("Local AI insights")
    st.caption(
        "Train scikit-learn models on the selected sales data—inside the app, "
        "without an API key or external AI request."
    )
    anomaly_tab, segment_tab, forecast_tab = st.tabs(
        ["ML Anomalies", "Market Segments", "ML Forecast"]
    )
    with anomaly_tab:
        _render_ml_anomalies(data)
    with segment_tab:
        _render_market_segments(data)
    with forecast_tab:
        _render_ml_forecast(data)
