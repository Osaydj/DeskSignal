"""Tests for CSV validation and cleaning."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

from desksignal.data.loader import load_sales_csv
from desksignal.data import validation
from desksignal.data.validation import validate_sales_data
from desksignal.errors import DataValidationError


def test_valid_data_is_cleaned(valid_sales_data: pd.DataFrame) -> None:
    cleaned = validate_sales_data(valid_sales_data)

    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    assert pd.api.types.is_integer_dtype(cleaned["units_sold"])
    assert len(cleaned) == 3


def test_column_names_are_normalized(valid_sales_data: pd.DataFrame) -> None:
    renamed = valid_sales_data.rename(
        columns={"transaction_id": "Transaction ID", "unit_price": "Unit Price"}
    )

    cleaned = validate_sales_data(renamed)

    assert "transaction_id" in cleaned.columns
    assert "unit_price" in cleaned.columns


def test_missing_columns_are_reported(valid_sales_data: pd.DataFrame) -> None:
    incomplete = valid_sales_data.drop(columns=["profit", "region"])

    with pytest.raises(DataValidationError) as error:
        validate_sales_data(incomplete)

    assert "profit" in str(error.value)
    assert "region" in str(error.value)


def test_invalid_dates_and_numbers_are_reported(
    valid_sales_data: pd.DataFrame,
) -> None:
    invalid = valid_sales_data.copy()
    invalid.loc[0, "date"] = "not-a-date"
    invalid["revenue"] = invalid["revenue"].astype("object")
    invalid.loc[1, "revenue"] = "not-a-number"

    with pytest.raises(DataValidationError) as error:
        validate_sales_data(invalid)

    assert any("invalid date" in message for message in error.value.messages)
    assert any("invalid number" in message for message in error.value.messages)


def test_empty_csv_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        load_sales_csv(StringIO(""))


def test_inconsistent_revenue_and_profit_are_rejected(
    valid_sales_data: pd.DataFrame,
) -> None:
    inconsistent = valid_sales_data.copy()
    inconsistent.loc[0, "revenue"] = 999.0
    inconsistent.loc[1, "profit"] = 999.0

    with pytest.raises(DataValidationError) as error:
        validate_sales_data(inconsistent)

    assert any("units_sold" in message for message in error.value.messages)
    assert any("revenue − cost" in message for message in error.value.messages)


def test_currency_rounding_within_one_cent_is_accepted(
    valid_sales_data: pd.DataFrame,
) -> None:
    rounded = valid_sales_data.copy()
    rounded.loc[0, "revenue"] += 0.01
    rounded.loc[0, "profit"] += 0.01

    cleaned = validate_sales_data(rounded)

    assert len(cleaned) == len(rounded)


def test_mixed_timezone_dates_are_normalized(
    valid_sales_data: pd.DataFrame,
) -> None:
    mixed = valid_sales_data.copy()
    mixed.loc[0, "date"] = "2025-01-01T00:00:00Z"
    mixed.loc[1, "date"] = "2025-01-02T00:00:00-05:00"

    cleaned = validate_sales_data(mixed)

    assert pd.api.types.is_datetime64_ns_dtype(cleaned["date"])
    assert cleaned["date"].dt.tz is None


def test_ambiguous_dates_are_rejected(valid_sales_data: pd.DataFrame) -> None:
    ambiguous = valid_sales_data.copy()
    ambiguous.loc[0, "date"] = "01/02/2025"

    with pytest.raises(DataValidationError, match="ambiguous"):
        validate_sales_data(ambiguous)


def test_row_limit_is_enforced(
    valid_sales_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "MAX_SALES_ROWS", 2)

    with pytest.raises(DataValidationError, match="at most 2 rows"):
        validate_sales_data(valid_sales_data)
