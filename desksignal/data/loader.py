"""CSV loading functions for DeskSignal."""

from __future__ import annotations

from os import PathLike
from typing import IO

import pandas as pd

from desksignal.data.validation import validate_sales_data
from desksignal.errors import DataValidationError


CsvSource = str | PathLike[str] | IO[bytes] | IO[str]


def load_sales_csv(source: CsvSource) -> pd.DataFrame:
    """Read a CSV source and return validated sales data."""

    try:
        raw_data = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The uploaded CSV is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(
            "The file could not be parsed as CSV. Check its commas and row structure."
        ) from exc
    except UnicodeDecodeError as exc:
        raise DataValidationError(
            "The CSV text encoding is not supported. Save it as UTF-8 and try again."
        ) from exc
    except (OSError, ValueError) as exc:
        raise DataValidationError("The uploaded CSV could not be read.") from exc

    return validate_sales_data(raw_data)
