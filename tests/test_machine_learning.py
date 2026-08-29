"""Tests for the local, no-API machine-learning analytics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from desksignal.analytics.machine_learning import (
    cluster_market_segments,
    create_ml_forecast,
    detect_ml_daily_anomalies,
)


@pytest.fixture(scope="module")
def sample_sales_data() -> pd.DataFrame:
    """Load the validated fictional portfolio dataset."""

    sample_path = (
        Path(__file__).parents[1] / "sample_data" / "sample_sales_data.csv"
    )
    return pd.read_csv(sample_path)


def test_isolation_forest_returns_ranked_daily_results(
    sample_sales_data: pd.DataFrame,
) -> None:
    results = detect_ml_daily_anomalies(sample_sales_data, contamination=0.04)

    assert len(results) == 365
    assert results["date"].is_monotonic_increasing
    assert 10 <= int(results["is_anomaly"].sum()) <= 20
    assert results["anomaly_score"].between(0, 100).all()
    assert results.loc[results["is_anomaly"], "anomaly_score"].mean() > 90


def test_isolation_forest_validates_settings_and_history(
    sample_sales_data: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="contamination"):
        detect_ml_daily_anomalies(sample_sales_data, contamination=0.50)

    short_history = sample_sales_data.loc[
        pd.to_datetime(sample_sales_data["date"]) < "2025-01-20"
    ]
    with pytest.raises(ValueError, match="at least 30"):
        detect_ml_daily_anomalies(short_history)


def test_kmeans_assigns_every_product_market_segment(
    sample_sales_data: pd.DataFrame,
) -> None:
    result = cluster_market_segments(sample_sales_data, n_clusters=4)

    assert result.n_clusters == 4
    assert set(result.assignments["cluster"]) == {1, 2, 3, 4}
    assert len(result.summary) == 4
    assert result.assignments["profile"].notna().all()
    assert -1 <= result.silhouette_score <= 1
    assert result.summary["revenue"].sum() == pytest.approx(
        sample_sales_data["revenue"].sum()
    )


def test_kmeans_is_reproducible(sample_sales_data: pd.DataFrame) -> None:
    first = cluster_market_segments(sample_sales_data, n_clusters=3)
    second = cluster_market_segments(sample_sales_data, n_clusters=3)

    pd.testing.assert_series_equal(
        first.assignments["cluster"],
        second.assignments["cluster"],
    )


def test_kmeans_rejects_too_many_clusters(
    valid_sales_data: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="more distinct"):
        cluster_market_segments(valid_sales_data, n_clusters=4)


def test_random_forest_forecast_includes_validation_evidence(
    sample_sales_data: pd.DataFrame,
) -> None:
    result = create_ml_forecast(
        sample_sales_data,
        metric="revenue",
        horizon_days=14,
    )

    assert result.model_name == "Random Forest"
    assert len(result.forecast) == 14
    assert (result.forecast["forecast"] >= 0).all()
    assert (result.forecast["lower_bound"] >= 0).all()
    assert result.mean_absolute_error >= 0
    assert result.weighted_absolute_percentage_error >= 0
    assert result.baseline_mean_absolute_error >= 0
    assert result.feature_importance["importance"].sum() == pytest.approx(1.0)
    assert result.validation["date"].is_monotonic_increasing


def test_random_forest_forecast_validates_metric_and_history(
    sample_sales_data: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="metric must be"):
        create_ml_forecast(sample_sales_data, metric="cost")

    short_history = sample_sales_data.loc[
        pd.to_datetime(sample_sales_data["date"]) < "2025-03-01"
    ]
    with pytest.raises(ValueError, match="at least 90"):
        create_ml_forecast(short_history, metric="profit")
