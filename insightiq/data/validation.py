"""Reusable validation and cleaning for uploaded sales data."""

from __future__ import annotations

import re

import pandas as pd

from insightiq.data.schema import (
    FINANCIAL_TOLERANCE,
    MAX_SALES_ROWS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TEXT_COLUMNS,
)
from insightiq.errors import DataValidationError


def normalize_column_name(column: object) -> str:
    """Convert a column label such as 'Unit Price' into 'unit_price'."""

    normalized = str(column).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def validate_sales_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a cleaned copy of an uploaded sales DataFrame.

    Extra columns are preserved, but every required column must be present and
    usable. The function collects multiple problems so the user can fix them in
    one edit instead of discovering errors one at a time.
    """

    if raw_data.empty:
        raise DataValidationError("The CSV contains no sales rows.")
    if len(raw_data) > MAX_SALES_ROWS:
        raise DataValidationError(
            f"The CSV contains {len(raw_data):,} rows. "
            f"InsightIQ accepts at most {MAX_SALES_ROWS:,} rows per upload."
        )

    data = raw_data.copy()
    normalized_columns = [normalize_column_name(column) for column in data.columns]

    duplicate_columns = sorted(
        {column for column in normalized_columns if normalized_columns.count(column) > 1}
    )
    if duplicate_columns:
        raise DataValidationError(
            "These column names become duplicates after normalization: "
            + ", ".join(duplicate_columns)
        )

    data.columns = normalized_columns
    missing_columns = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing_columns:
        raise DataValidationError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    errors: list[str] = []

    for column in TEXT_COLUMNS:
        cleaned = data[column].astype("string").str.strip()
        blank_mask = cleaned.isna() | cleaned.eq("")
        if blank_mask.any():
            errors.append(
                f"Column '{column}' contains {int(blank_mask.sum())} blank value(s)."
            )
        data[column] = cleaned

    raw_dates = data["date"].astype("string").str.strip()
    ambiguous_dates = raw_dates.str.match(
        r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?:\s|$)",
        na=False,
    )
    if ambiguous_dates.any():
        errors.append(
            "Column 'date' contains "
            f"{int(ambiguous_dates.sum())} ambiguous date value(s). "
            "Use ISO format YYYY-MM-DD."
        )

    parsed_dates = pd.to_datetime(
        raw_dates.mask(ambiguous_dates),
        errors="coerce",
        format="mixed",
        utc=True,
    )
    invalid_dates = parsed_dates.isna() & ~ambiguous_dates
    if invalid_dates.any():
        errors.append(
            f"Column 'date' contains {int(invalid_dates.sum())} invalid date value(s)."
        )
    # Normalize timezone-aware and timezone-naive ISO values to one comparable,
    # timezone-free business-date representation.
    data["date"] = parsed_dates.dt.tz_convert(None).dt.normalize()

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid_numbers = converted.isna()
        if invalid_numbers.any():
            errors.append(
                f"Column '{column}' contains "
                f"{int(invalid_numbers.sum())} invalid number(s)."
            )
        infinite_numbers = converted.isin([float("inf"), float("-inf")])
        if infinite_numbers.any():
            errors.append(f"Column '{column}' contains infinite values.")
        data[column] = converted

    valid_units = data["units_sold"].dropna()
    if (valid_units <= 0).any():
        errors.append("Column 'units_sold' must contain values greater than zero.")
    if ((valid_units % 1) != 0).any():
        errors.append("Column 'units_sold' must contain whole numbers.")

    valid_prices = data["unit_price"].dropna()
    if (valid_prices <= 0).any():
        errors.append("Column 'unit_price' must contain values greater than zero.")

    for column in ("revenue", "cost"):
        if (data[column].dropna() < 0).any():
            errors.append(f"Column '{column}' cannot contain negative values.")

    revenue_inputs_are_valid = data[
        ["units_sold", "unit_price", "revenue"]
    ].notna().all(axis=1)
    expected_revenue = data["units_sold"] * data["unit_price"]
    inconsistent_revenue = revenue_inputs_are_valid & (
        (data["revenue"] - expected_revenue).abs() > FINANCIAL_TOLERANCE
    )
    if inconsistent_revenue.any():
        errors.append(
            "Column 'revenue' does not equal units_sold × unit_price within "
            f"one cent for {int(inconsistent_revenue.sum())} row(s)."
        )

    profit_inputs_are_valid = data[["revenue", "cost", "profit"]].notna().all(axis=1)
    expected_profit = data["revenue"] - data["cost"]
    inconsistent_profit = profit_inputs_are_valid & (
        (data["profit"] - expected_profit).abs() > FINANCIAL_TOLERANCE
    )
    if inconsistent_profit.any():
        errors.append(
            "Column 'profit' does not equal revenue − cost within one cent for "
            f"{int(inconsistent_profit.sum())} row(s)."
        )

    if errors:
        raise DataValidationError(errors)

    data["units_sold"] = data["units_sold"].astype("int64")
    return data.sort_values("date", ignore_index=True)
