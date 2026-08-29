"""Local, deterministic machine-learning models for business analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, silhouette_score
from sklearn.preprocessing import StandardScaler


SUPPORTED_METRICS: Final = frozenset({"revenue", "profit"})
MINIMUM_ANOMALY_DAYS: Final = 30
MINIMUM_FORECAST_DAYS: Final = 90
MAXIMUM_FORECAST_HORIZON: Final = 90
MAXIMUM_TRAINING_DAYS: Final = 730
SEGMENT_KEYS: Final = ("product", "category", "region", "customer_type")
SEGMENT_FEATURES: Final = (
    "revenue",
    "profit",
    "units_sold",
    "transactions",
    "average_order_value",
    "profit_margin",
)
FORECAST_FEATURES: Final = (
    "trend_index",
    "day_of_week",
    "month",
    "day_of_year_sin",
    "day_of_year_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
)


@dataclass(frozen=True)
class MarketSegmentation:
    """Store K-Means assignments, summaries, and model quality."""

    assignments: pd.DataFrame
    summary: pd.DataFrame
    n_clusters: int
    silhouette_score: float


@dataclass(frozen=True)
class MLForecast:
    """Store a local random-forest forecast and validation evidence."""

    metric: str
    history: pd.DataFrame
    forecast: pd.DataFrame
    validation: pd.DataFrame
    horizon_days: int
    mean_absolute_error: float
    weighted_absolute_percentage_error: float
    baseline_mean_absolute_error: float
    baseline_weighted_absolute_percentage_error: float
    improvement_over_baseline: float
    feature_importance: pd.DataFrame
    model_name: str = "Random Forest"


def _require_columns(data: pd.DataFrame, required: set[str]) -> None:
    """Raise a readable error when model inputs are incomplete."""

    missing_columns = required.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}.")


def _clean_daily_sales(data: pd.DataFrame) -> pd.DataFrame:
    """Return one validated row per calendar date for local models."""

    required = {
        "date",
        "transaction_id",
        "revenue",
        "profit",
        "units_sold",
    }
    _require_columns(data, required)
    if data.empty:
        raise ValueError("Machine-learning analysis requires a non-empty dataset.")

    cleaned = data.loc[:, sorted(required)].copy()
    cleaned["date"] = pd.to_datetime(
        cleaned["date"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    for column in ("revenue", "profit", "units_sold"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if cleaned[["date", "revenue", "profit", "units_sold"]].isna().any().any():
        raise ValueError("Machine-learning inputs contain invalid dates or numbers.")

    cleaned["date"] = cleaned["date"].dt.normalize()
    daily = (
        cleaned.groupby("date", as_index=True)
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            units_sold=("units_sold", "sum"),
            transactions=("transaction_id", "nunique"),
        )
        .sort_index()
        .astype(float)
    )

    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar, fill_value=0.0)
    daily.index.name = "date"
    daily["average_order_value"] = daily["revenue"].divide(
        daily["transactions"].replace(0, np.nan)
    ).fillna(0.0)
    daily["profit_margin"] = daily["profit"].divide(
        daily["revenue"].replace(0, np.nan)
    ).fillna(0.0)
    return daily


def detect_ml_daily_anomalies(
    data: pd.DataFrame,
    *,
    contamination: float = 0.04,
    random_state: int = 42,
) -> pd.DataFrame:
    """Flag unusual days with a multivariate Isolation Forest model.

    Revenue, profit, units, transaction count, average order value, and margin
    are evaluated together. The score is a within-selection percentile where a
    larger value means the day is more unusual relative to the selected data.
    """

    if not 0.01 <= contamination <= 0.20:
        raise ValueError("contamination must be between 0.01 and 0.20.")

    daily = _clean_daily_sales(data)
    if len(daily) < MINIMUM_ANOMALY_DAYS:
        raise ValueError(
            "ML anomaly detection requires at least "
            f"{MINIMUM_ANOMALY_DAYS} calendar days of history."
        )

    feature_columns = list(SEGMENT_FEATURES)
    scaled_features = StandardScaler().fit_transform(daily[feature_columns])
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,
    )
    predictions = model.fit_predict(scaled_features)
    raw_scores = -model.decision_function(scaled_features)

    results = daily.reset_index()
    results["model_score"] = raw_scores
    results["anomaly_score"] = (
        pd.Series(raw_scores).rank(method="average", pct=True) * 100
    )
    results["is_anomaly"] = predictions == -1

    typical_revenue = float(results["revenue"].median())
    results["direction"] = np.where(
        results["revenue"] >= typical_revenue,
        "Above typical revenue",
        "Below typical revenue",
    )
    results.loc[~results["is_anomaly"], "direction"] = "Typical"

    return results.loc[
        :,
        [
            "date",
            "revenue",
            "profit",
            "units_sold",
            "transactions",
            "average_order_value",
            "profit_margin",
            "model_score",
            "anomaly_score",
            "direction",
            "is_anomaly",
        ],
    ]


def _signed_log(values: pd.Series) -> pd.Series:
    """Compress large ranges while preserving negative profit values."""

    numeric = values.astype(float)
    return np.sign(numeric) * np.log1p(np.abs(numeric))


def _segment_profile(
    *,
    average_revenue: float,
    profit_margin: float,
    revenue_reference: float,
    margin_reference: float,
) -> str:
    """Assign a cautious, descriptive label to one market cluster."""

    high_revenue = average_revenue >= revenue_reference
    high_margin = profit_margin >= margin_reference

    if high_revenue and high_margin:
        return "Revenue & margin leaders"
    if high_revenue:
        return "High-volume drivers"
    if high_margin:
        return "High-margin niches"
    return "Growth watch"


def cluster_market_segments(
    data: pd.DataFrame,
    *,
    n_clusters: int = 4,
    random_state: int = 42,
) -> MarketSegmentation:
    """Group product-market combinations with deterministic K-Means."""

    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
        raise ValueError("n_clusters must be a whole number.")
    if not 2 <= n_clusters <= 8:
        raise ValueError("n_clusters must be between 2 and 8.")

    required = set(SEGMENT_KEYS).union(
        {"transaction_id", "revenue", "profit", "units_sold"}
    )
    _require_columns(data, required)
    if data.empty:
        raise ValueError("Market segmentation requires a non-empty dataset.")

    segments = (
        data.groupby(list(SEGMENT_KEYS), as_index=False, observed=True)
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            units_sold=("units_sold", "sum"),
            transactions=("transaction_id", "nunique"),
        )
        .sort_values(list(SEGMENT_KEYS), ignore_index=True)
    )
    segments["average_order_value"] = segments["revenue"].divide(
        segments["transactions"].replace(0, np.nan)
    ).fillna(0.0)
    segments["profit_margin"] = segments["profit"].divide(
        segments["revenue"].replace(0, np.nan)
    ).fillna(0.0)

    if len(segments) <= n_clusters:
        raise ValueError(
            "Market segmentation needs more distinct product-market segments "
            "than the selected number of clusters."
        )

    model_features = segments.loc[:, list(SEGMENT_FEATURES)].copy()
    for column in SEGMENT_FEATURES[:-1]:
        model_features[column] = _signed_log(model_features[column])
    scaled_features = StandardScaler().fit_transform(model_features)

    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    original_labels = model.fit_predict(scaled_features)

    segments["_original_cluster"] = original_labels
    revenue_order = (
        segments.groupby("_original_cluster", observed=True)["revenue"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    cluster_map = {
        original_cluster: display_cluster
        for display_cluster, original_cluster in enumerate(revenue_order, start=1)
    }
    segments["cluster"] = segments["_original_cluster"].map(cluster_map).astype(int)

    revenue_reference = float(segments["revenue"].median())
    margin_reference = float(segments["profit_margin"].median())
    summary_rows: list[dict[str, object]] = []

    for cluster, cluster_data in segments.groupby("cluster", sort=True):
        total_revenue = float(cluster_data["revenue"].sum())
        total_profit = float(cluster_data["profit"].sum())
        transactions = int(cluster_data["transactions"].sum())
        units_sold = float(cluster_data["units_sold"].sum())
        average_revenue = float(cluster_data["revenue"].mean())
        profit_margin = total_profit / total_revenue if total_revenue else 0.0
        profile = _segment_profile(
            average_revenue=average_revenue,
            profit_margin=profit_margin,
            revenue_reference=revenue_reference,
            margin_reference=margin_reference,
        )
        summary_rows.append(
            {
                "cluster": int(cluster),
                "profile": profile,
                "segment_count": int(len(cluster_data)),
                "revenue": total_revenue,
                "profit": total_profit,
                "profit_margin": float(profit_margin),
                "transactions": transactions,
                "units_sold": units_sold,
            }
        )

    summary = pd.DataFrame(summary_rows)
    profile_map = summary.set_index("cluster")["profile"].to_dict()
    segments["profile"] = segments["cluster"].map(profile_map)
    segments = segments.drop(columns="_original_cluster")

    quality_score = float(silhouette_score(scaled_features, original_labels))
    return MarketSegmentation(
        assignments=segments,
        summary=summary,
        n_clusters=n_clusters,
        silhouette_score=quality_score,
    )


def _daily_metric_series(data: pd.DataFrame, metric: str) -> pd.Series:
    """Return a complete daily series for one supported forecast metric."""

    if metric not in SUPPORTED_METRICS:
        supported = ", ".join(sorted(SUPPORTED_METRICS))
        raise ValueError(f"metric must be one of: {supported}.")

    _require_columns(data, {"date", metric})
    if data.empty:
        raise ValueError("ML forecasting requires a non-empty dataset.")

    cleaned = data.loc[:, ["date", metric]].copy()
    cleaned["date"] = pd.to_datetime(
        cleaned["date"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    cleaned[metric] = pd.to_numeric(cleaned[metric], errors="coerce")
    if cleaned.isna().any().any():
        raise ValueError("ML forecasting inputs contain invalid dates or numbers.")

    cleaned["date"] = cleaned["date"].dt.normalize()
    daily = cleaned.groupby("date", observed=True)[metric].sum().sort_index()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(calendar, fill_value=0.0).astype(float).tail(
        MAXIMUM_TRAINING_DAYS
    )


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build deterministic calendar features for daily forecasting."""

    day_of_year = index.dayofyear.astype(float)
    return pd.DataFrame(
        {
            "day_of_week": index.dayofweek.astype(float),
            "month": index.month.astype(float),
            "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        },
        index=index,
    )


def _supervised_forecast_frame(series: pd.Series) -> pd.DataFrame:
    """Create chronological lag and rolling features without target leakage."""

    frame = pd.DataFrame({"target": series.astype(float)})
    frame["trend_index"] = np.arange(len(frame), dtype=float)
    calendar = _calendar_features(frame.index)
    for column in calendar.columns:
        frame[column] = calendar[column]

    for lag in (1, 7, 14, 28):
        frame[f"lag_{lag}"] = frame["target"].shift(lag)
    shifted = frame["target"].shift(1)
    frame["rolling_mean_7"] = shifted.rolling(7).mean()
    frame["rolling_mean_28"] = shifted.rolling(28).mean()
    frame["rolling_std_7"] = shifted.rolling(7).std(ddof=0)
    return frame.dropna()


def _future_feature_row(series: pd.Series, future_date: pd.Timestamp) -> pd.DataFrame:
    """Build one recursive forecast row from known and predicted history."""

    calendar = _calendar_features(pd.DatetimeIndex([future_date])).iloc[0]
    values = {
        "trend_index": float(len(series)),
        "day_of_week": float(calendar["day_of_week"]),
        "month": float(calendar["month"]),
        "day_of_year_sin": float(calendar["day_of_year_sin"]),
        "day_of_year_cos": float(calendar["day_of_year_cos"]),
        "lag_1": float(series.iloc[-1]),
        "lag_7": float(series.iloc[-7]),
        "lag_14": float(series.iloc[-14]),
        "lag_28": float(series.iloc[-28]),
        "rolling_mean_7": float(series.iloc[-7:].mean()),
        "rolling_mean_28": float(series.iloc[-28:].mean()),
        "rolling_std_7": float(series.iloc[-7:].std(ddof=0)),
    }
    return pd.DataFrame([values], index=[future_date]).loc[:, list(FORECAST_FEATURES)]


def _wape(actual: pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Return weighted absolute percentage error as a percentage."""

    actual_values = actual.to_numpy(dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    denominator = float(np.abs(actual_values).sum())
    if denominator == 0:
        return 0.0
    return float(np.abs(actual_values - predicted_values).sum() / denominator * 100)


def create_ml_forecast(
    data: pd.DataFrame,
    *,
    metric: str = "revenue",
    horizon_days: int = 30,
    random_state: int = 42,
) -> MLForecast:
    """Forecast a daily metric with a locally trained random forest.

    The newest observations are held out chronologically for validation. Model
    error is compared with a transparent seven-day-lag baseline before the model
    is retrained on all available supervised rows for the future forecast.
    """

    if (
        isinstance(horizon_days, bool)
        or not isinstance(horizon_days, int)
        or not 1 <= horizon_days <= MAXIMUM_FORECAST_HORIZON
    ):
        raise ValueError(
            f"horizon_days must be between 1 and {MAXIMUM_FORECAST_HORIZON}."
        )

    daily = _daily_metric_series(data, metric)
    if len(daily) < MINIMUM_FORECAST_DAYS:
        raise ValueError(
            "ML forecasting requires at least "
            f"{MINIMUM_FORECAST_DAYS} calendar days of history."
        )

    supervised = _supervised_forecast_frame(daily)
    validation_size = min(30, max(14, int(len(supervised) * 0.20)))
    if len(supervised) - validation_size < 30:
        raise ValueError("ML forecasting needs more history for validation.")

    training = supervised.iloc[:-validation_size]
    validation = supervised.iloc[-validation_size:]
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(training.loc[:, list(FORECAST_FEATURES)], training["target"])
    validation_predictions = model.predict(
        validation.loc[:, list(FORECAST_FEATURES)]
    )
    if metric == "revenue":
        validation_predictions = np.clip(validation_predictions, 0.0, None)

    baseline_predictions = validation["lag_7"].to_numpy(dtype=float)
    validation_actual = validation["target"]
    model_mae = float(mean_absolute_error(validation_actual, validation_predictions))
    baseline_mae = float(mean_absolute_error(validation_actual, baseline_predictions))
    model_wape = _wape(validation_actual, validation_predictions)
    baseline_wape = _wape(validation_actual, baseline_predictions)
    improvement = (
        ((baseline_mae - model_mae) / baseline_mae) * 100
        if baseline_mae > 0
        else 0.0
    )

    validation_frame = pd.DataFrame(
        {
            "date": validation.index,
            "actual": validation_actual.to_numpy(dtype=float),
            "prediction": validation_predictions,
            "seven_day_baseline": baseline_predictions,
        }
    )
    absolute_errors = np.abs(
        validation_frame["actual"] - validation_frame["prediction"]
    )
    uncertainty = float(np.quantile(absolute_errors, 0.90))

    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )
    final_model.fit(
        supervised.loc[:, list(FORECAST_FEATURES)], supervised["target"]
    )

    recursive_history = daily.copy()
    future_rows: list[dict[str, object]] = []
    for _ in range(horizon_days):
        future_date = recursive_history.index.max() + pd.DateOffset(days=1)
        future_features = _future_feature_row(recursive_history, future_date)
        predicted = float(final_model.predict(future_features)[0])
        if metric == "revenue":
            predicted = max(0.0, predicted)

        lower_bound = predicted - uncertainty
        if metric == "revenue":
            lower_bound = max(0.0, lower_bound)
        future_rows.append(
            {
                "date": future_date,
                "forecast": predicted,
                "lower_bound": lower_bound,
                "upper_bound": predicted + uncertainty,
            }
        )
        recursive_history.loc[future_date] = predicted

    importance = (
        pd.DataFrame(
            {
                "feature": list(FORECAST_FEATURES),
                "importance": final_model.feature_importances_,
            }
        )
        .sort_values(["importance", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    history = daily.rename("actual").rename_axis("date").reset_index()
    return MLForecast(
        metric=metric,
        history=history,
        forecast=pd.DataFrame(future_rows),
        validation=validation_frame,
        horizon_days=horizon_days,
        mean_absolute_error=model_mae,
        weighted_absolute_percentage_error=model_wape,
        baseline_mean_absolute_error=baseline_mae,
        baseline_weighted_absolute_percentage_error=baseline_wape,
        improvement_over_baseline=float(improvement),
        feature_importance=importance,
    )
