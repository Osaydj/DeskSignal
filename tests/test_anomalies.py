"""Tests for deterministic daily anomaly detection."""

from __future__ import annotations

import pandas as pd
import pytest

from insightiq.analytics.anomalies import detect_daily_anomalies
from insightiq.visualization.charts import daily_anomaly_figure

def test_known_revenue_spike_and_drop_are_detected() -> None:
    dates = pd.date_range("2025-01-01", periods=60)
    revenue = [100.0 + (index % 5) for index in range(60)]
    revenue[30] = 1_000.0
    revenue[45] = 5.0
    data = pd.DataFrame({"date": dates, "revenue": revenue})

    result = detect_daily_anomalies(data)
    anomalies = result.loc[result["is_anomaly"]]

    assert len(result) == 60
    assert anomalies["date"].tolist() == [dates[30], dates[45]]
    assert anomalies["direction"].tolist() == ["Spike", "Drop"]
    assert anomalies["anomaly_score"].abs().min() >= 3.5


def test_transactions_are_aggregated_by_date() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-01-02"]
            ),
            "profit": [10.0, 15.0, 20.0],
        }
    )

    result = detect_daily_anomalies(
        data,
        metric="profit",
        threshold=100.0,
    )

    assert len(result) == 2
    assert result["actual"].tolist() == [25.0, 20.0]
    assert result["metric"].unique().tolist() == ["profit"]
    assert not result["is_anomaly"].any()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"metric": "units_sold"}, "metric"),
        ({"window": 2}, "window"),
        ({"threshold": 0}, "threshold"),
    ],
)
def test_invalid_settings_are_rejected(
    options: dict[str, object],
    message: str,
) -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "revenue": [100.0],
        }
    )

    with pytest.raises(ValueError, match=message):
        detect_daily_anomalies(data, **options)


def test_missing_metric_column_is_rejected() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        detect_daily_anomalies(data)


def test_anomaly_chart_contains_actual_baseline_and_flags() -> None:
        dates = pd.date_range("2025-01-01", periods=40)
        revenue = [100.0 + (index % 5) for index in range(40)]
        revenue[20] = 1_000.0
        data = pd.DataFrame({"date": dates, "revenue": revenue})

        analysis = detect_daily_anomalies(data)
        figure = daily_anomaly_figure(analysis, metric="revenue")

        assert len(figure.data) == 3
        assert [trace.name for trace in figure.data] == [
            "Actual revenue",
            "Rolling baseline",
            "Flagged day",
        ]
        assert len(figure.data[2].x) == 1
        assert figure.layout.title.text == "Daily revenue anomaly review"