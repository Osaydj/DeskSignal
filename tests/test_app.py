"""Basic render test for the Streamlit entry point."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import streamlit_app
from desksignal.errors import DataValidationError


def test_initial_page_renders_without_exceptions() -> None:
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=10).run()

    assert len(app.exception) == 0
    assert app.title[0].value.startswith("DeskSignal")
    assert len(app.info) == 1


def test_sample_upload_renders_dashboard_and_local_ml_tools() -> None:
    project_root = Path(__file__).parents[1]
    app = AppTest.from_file(
        project_root / "streamlit_app.py", default_timeout=20
    ).run()
    sample_bytes = (
        project_root / "sample_data" / "sample_sales_data.csv"
    ).read_bytes()

    app.file_uploader[0].upload(
        "sample_sales_data.csv", sample_bytes, "text/csv"
    ).run(timeout=20)

    assert len(app.exception) == 0
    assert len(app.success) == 1
    assert len(app.metric) == 26
    expected_tabs = {
        "Overview",
        "Trends",
        "Anomalies",
        "Forecasting",
        "Products & Markets",
        "Data Explorer",
        "AI Insights",
    }
    assert expected_tabs.issubset({tab.label for tab in app.tabs})
    assert {"ML Anomalies", "Market Segments", "ML Forecast"}.issubset(
        {tab.label for tab in app.tabs}
    )
    assert not any(
        button.label == "Generate AI Executive Summary" for button in app.button
    )
    assert not any(
        button.label == "Generate AI Recommendations" for button in app.button
    )
    assert len(app.chat_input) == 0
    assert any(
        "local machine-learning tools" in message.value
        for message in app.info
    )
    assert any(
        button.label == "Download filtered CSV"
        for button in app.get("download_button")
    )


def test_ai_sections_can_be_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_FEATURES_ENABLED", "true")
    project_root = Path(__file__).parents[1]
    app = AppTest.from_file(
        project_root / "streamlit_app.py", default_timeout=20
    ).run()
    sample_bytes = (
        project_root / "sample_data" / "sample_sales_data.csv"
    ).read_bytes()

    app.file_uploader[0].upload(
        "sample_sales_data.csv", sample_bytes, "text/csv"
    ).run(timeout=20)

    assert len(app.exception) == 0
    assert any(
        button.label == "Generate AI Executive Summary" for button in app.button
    )
    assert any(
        button.label == "Generate AI Recommendations" for button in app.button
    )
    assert len(app.chat_input) == 1


def test_server_side_upload_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streamlit_app, "MAX_UPLOAD_BYTES", 3)
    monkeypatch.setattr(streamlit_app, "MAX_UPLOAD_SIZE_MB", 1)

    with pytest.raises(DataValidationError, match="1 MB"):
        streamlit_app.validate_upload_size(4)


def test_sample_data_button_renders_dashboard() -> None:
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()

    sample_button = next(
        button for button in app.button if button.label == "Try it with sample data"
    )
    sample_button.click().run(timeout=20)

    assert len(app.exception) == 0
    assert any("fictional sample data" in message.value for message in app.success)
    assert any(
        button.label == "Download filtered CSV"
        for button in app.get("download_button")
    )
