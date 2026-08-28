"""Streamlit entry point for the InsightIQ analytics dashboard."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from insightiq.config import APP_ICON, APP_PHASE, APP_TITLE
from insightiq.data.loader import load_sales_csv
from insightiq.errors import DataValidationError
from insightiq.ui.dashboard import render_dashboard


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
        page_title="InsightIQ",
        page_icon=APP_ICON,
        layout="wide",
    )

    st.title(APP_TITLE)
    st.caption(APP_PHASE)
    st.write(
        "Upload fictional sales data to calculate KPIs and explore interactive "
        "business trends. Python calculates every statistic; Claude can optionally "
        "interpret those results and retrieve relevant sections from a fictional "
        "company knowledge base to propose cautious investigation ideas."
    )

    uploaded_file = st.file_uploader(
        "Upload a sales CSV",
        type=["csv"],
        help=(
            "The file must contain the required InsightIQ sales columns and use "
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
