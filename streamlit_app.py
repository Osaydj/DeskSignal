"""Streamlit entry point for the DeskSignal analytics dashboard."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from desksignal.config import (
    APP_ICON,
    APP_PHASE,
    APP_TAGLINE,
    APP_TITLE,
    ai_features_enabled,
)
from desksignal.data.loader import load_sales_csv
from desksignal.errors import DataValidationError
from desksignal.ui.dashboard import render_dashboard


LOGGER = logging.getLogger(__name__)
SAMPLE_DATA_PATH = Path(__file__).parent / "sample_data" / "sample_sales_data.csv"
MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def validate_upload_size(file_size: int) -> None:
    """Reject files that exceed the server-side upload limit."""

    if file_size > MAX_UPLOAD_BYTES:
        raise DataValidationError(
            f"The CSV is larger than the {MAX_UPLOAD_SIZE_MB} MB upload limit."
        )


@st.cache_data(show_spinner=False, ttl=1_800, max_entries=8)
def load_uploaded_data(file_bytes: bytes) -> pd.DataFrame:
    """Load and validate uploaded bytes, caching identical uploads."""

    validate_upload_size(len(file_bytes))
    return load_sales_csv(BytesIO(file_bytes))


def render_sample_download() -> None:
    """Offer the bundled fictional dataset when it is available."""

    if not SAMPLE_DATA_PATH.exists():
        return

    st.download_button(
        label="Download the fictional sample CSV",
        data=SAMPLE_DATA_PATH.read_bytes(),
        file_name="sample_sales_data.csv",
        mime="text/csv",
    )


def main() -> None:
    """Render the analytics application and optional AI summary."""

    st.set_page_config(
        page_title=f"{APP_TITLE} | Business Analytics",
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                "DeskSignal is a portfolio and educational business analytics "
                "application created by Osayd Jahanzeb."
            ),
        },
    )

    st.title(APP_TITLE)
    st.caption(f"{APP_TAGLINE} · © 2026 Osayd Jahanzeb. All rights reserved.")

    with st.container(horizontal=True, gap="small"):
        st.badge(
            APP_PHASE,
            icon=":material/analytics:",
            color="primary",
        )
        st.badge(
            "Python-calculated results",
            icon=":material/verified:",
            color="green",
        )
        if not ai_features_enabled():
            st.badge(
                "AI disabled by default",
                icon=":material/lock:",
                color="gray",
            )

    st.write(
        "Upload fictional sales data to explore KPIs, trends, anomalies, forecasts, "
        "and product performance. Python calculates every displayed statistic. "
        "Claude features remain optional and operator-controlled."
    )

    uploaded_file = st.file_uploader(
        "Upload a sales CSV",
        type=["csv"],
        help=(
            "The file must contain the required DeskSignal sales columns and use "
            "YYYY-MM-DD dates. Maximum file size: 10 MB."
        ),
        max_upload_size=MAX_UPLOAD_SIZE_MB,
    )

    if uploaded_file is None:
        st.info("Upload a CSV to begin. You can use the fictional sample below.")
        render_sample_download()
        return

    try:
        sales_data = load_uploaded_data(uploaded_file.getvalue())
    except DataValidationError as exc:
        st.error("The CSV could not be processed. Please fix these issues:")
        for message in exc.messages:
            st.markdown(f"- {message}")
        return
    except Exception:
        LOGGER.exception("Unexpected error while processing an uploaded CSV")
        st.error(
            "An unexpected error occurred while reading the CSV. "
            "Check the file and try again."
        )
        return

    st.success(f"Validated {len(sales_data):,} sales rows successfully.")
    render_dashboard(sales_data)


if __name__ == "__main__":
    main()
