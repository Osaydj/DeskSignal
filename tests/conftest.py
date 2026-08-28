"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def valid_sales_data() -> pd.DataFrame:
    """Return a small valid line-item dataset with known answers."""

    return pd.DataFrame(
        {
            "transaction_id": ["TXN-001", "TXN-002", "TXN-002"],
            "date": ["2025-01-01", "2025-01-02", "2025-01-02"],
            "product": ["Alpha", "Beta", "Alpha"],
            "category": ["Accessories", "Electronics", "Accessories"],
            "region": ["East", "West", "West"],
            "customer_type": ["Retail", "Business", "Business"],
            "units_sold": [2, 1, 3],
            "unit_price": [10.0, 30.0, 10.0],
            "revenue": [20.0, 30.0, 30.0],
            "cost": [12.0, 20.0, 18.0],
            "profit": [8.0, 10.0, 12.0],
        }
    )
