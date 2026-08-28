"""Tests for the deterministic data package sent to Claude."""

from __future__ import annotations

import json

import pandas as pd

from insightiq.ai.context import (
    build_analysis_context,
    context_fingerprint,
    evidence_lookup,
)
from insightiq.data.validation import validate_sales_data


def test_analysis_context_contains_aggregates_not_raw_rows(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    context = build_analysis_context(data)
    serialized = json.dumps(context)

    assert context["kpis"]["total_revenue"] == 80.0
    assert context["kpis"]["profit_margin_pct"] == 37.5
    assert context["category_performance"]["total_groups"] == 2
    assert context["product_performance"]["total_groups"] == 2
    assert context["regional_performance"]["total_groups"] == 2
    assert "TXN-001" not in serialized
    assert "customer_type" not in serialized


def test_context_evidence_is_unique_and_fingerprint_is_stable(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)
    first = build_analysis_context(data)
    second = build_analysis_context(data.copy())

    facts = evidence_lookup(first)

    assert facts["kpi.total_revenue"] == "Total revenue: $80.00."
    assert len(facts) == len(first["evidence_catalog"])
    assert context_fingerprint(first) == context_fingerprint(second)
