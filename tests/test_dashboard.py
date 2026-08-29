"""Tests for defensive dashboard input handling."""

from __future__ import annotations

from datetime import date

from desksignal.ui.dashboard import resolve_date_range


START = date(2025, 1, 1)
END = date(2025, 12, 31)


def test_empty_date_range_uses_full_dataset_range() -> None:
    assert resolve_date_range((), fallback_start=START, fallback_end=END) == (
        START,
        END,
    )


def test_partial_date_range_uses_the_selected_day() -> None:
    selected = date(2025, 7, 15)

    assert resolve_date_range(
        (selected,), fallback_start=START, fallback_end=END
    ) == (selected, selected)


def test_complete_date_range_is_ordered() -> None:
    assert resolve_date_range(
        (END, START), fallback_start=START, fallback_end=END
    ) == (START, END)
