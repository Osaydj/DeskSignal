"""Safe download helpers for user-facing CSV files."""

from __future__ import annotations

import pandas as pd


FORMULA_PREFIXES = ("=", "+", "-", "@")


def _escape_spreadsheet_formula(value: object) -> object:
    """Prevent text cells from being interpreted as spreadsheet formulas."""

    if isinstance(value, str) and value.lstrip().startswith(FORMULA_PREFIXES):
        return f"'{value}"
    return value


def dataframe_to_safe_csv(data: pd.DataFrame) -> bytes:
    """Serialize a DataFrame for Excel-friendly, formula-safe download."""

    safe_data = data.copy()
    text_columns = safe_data.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        safe_data[column] = safe_data[column].map(_escape_spreadsheet_formula)

    return safe_data.to_csv(index=False).encode("utf-8-sig")
