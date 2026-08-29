"""Shared display formatting for the Streamlit interface."""

from __future__ import annotations

from desksignal.config import CURRENCY_SYMBOL


def format_currency(value: float) -> str:
    """Format a number as U.S. currency with the sign before the symbol."""

    sign = "-" if value < 0 else ""
    return f"{sign}{CURRENCY_SYMBOL}{abs(value):,.2f}"
