"""Shared display formatting for the Streamlit interface."""

from __future__ import annotations

from desksignal.config import CURRENCY_SYMBOL


def format_currency(value: float) -> str:
    """Format a number as U.S. currency with the sign before the symbol."""

    sign = "-" if value < 0 else ""
    return f"{sign}{CURRENCY_SYMBOL}{abs(value):,.2f}"


def format_compact_currency(value: float) -> str:
    """Format large amounts compactly, such as $1.87M, for narrow metric tiles.

    Amounts under $1,000 keep full cents. Larger amounts use K, M, or B with
    three significant digits.
    """

    sign = "-" if value < 0 else ""
    amount = abs(value)
    units = (("B", 1e9), ("M", 1e6), ("K", 1e3))

    for index, (suffix, divisor) in enumerate(units):
        if amount < divisor:
            continue
        scaled = amount / divisor
        decimals = 2 if scaled < 10 else 1 if scaled < 100 else 0
        text = f"{scaled:.{decimals}f}"
        if float(text) >= 1_000 and index > 0:
            suffix, divisor = units[index - 1]
            text = f"{amount / divisor:.2f}"
        return f"{sign}{CURRENCY_SYMBOL}{text}{suffix}"

    return f"{sign}{CURRENCY_SYMBOL}{amount:,.2f}"
