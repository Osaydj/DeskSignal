"""Tests for shared Streamlit display and download helpers."""

from __future__ import annotations

from io import StringIO

import pandas as pd

from desksignal.ui.downloads import dataframe_to_safe_csv
from desksignal.ui.formatting import format_currency


def test_currency_sign_appears_before_symbol() -> None:
    assert format_currency(1_234.5) == "$1,234.50"
    assert format_currency(-1_234.5) == "-$1,234.50"
    assert format_currency(0.0) == "$0.00"


def test_csv_download_escapes_spreadsheet_formula_prefixes() -> None:
    data = pd.DataFrame(
        {
            "product": ["=SUM(1,1)", "+cmd", "-value", "@user", "Safe"],
            "revenue": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    csv_text = dataframe_to_safe_csv(data).decode("utf-8-sig")
    downloaded = pd.read_csv(StringIO(csv_text))

    assert downloaded["product"].tolist() == [
        "'=SUM(1,1)",
        "'+cmd",
        "'-value",
        "'@user",
        "Safe",
    ]
    assert downloaded["revenue"].tolist() == data["revenue"].tolist()
