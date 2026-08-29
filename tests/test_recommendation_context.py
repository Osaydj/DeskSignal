"""Tests for the calculated and retrieved recommendation evidence package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from desksignal.ai.recommendation_context import (
    MAX_RETRIEVED_SECTIONS,
    build_recommendation_context,
    recommendation_analytics_evidence,
    recommendation_context_fingerprint,
    recommendation_document_evidence,
)
from desksignal.data.validation import validate_sales_data


PROJECT_ROOT = Path(__file__).parents[1]


def test_recommendation_context_uses_aggregates_and_relevant_sections() -> None:
    data = validate_sales_data(
        pd.read_csv(PROJECT_ROOT / "sample_data" / "sample_sales_data.csv")
    )

    context = build_recommendation_context(data)
    serialized = json.dumps(context)
    documents = recommendation_document_evidence(context)

    assert context["analytics"]["kpis"]["total_revenue"] > 0
    assert 1 <= len(documents) <= MAX_RETRIEVED_SECTIONS
    assert len(documents) == context["company_context"]["retrieved_section_count"]
    assert all(
        source["source_path"].startswith(
            ("policies/", "products/", "strategy/")
        )
        for source in documents.values()
    )
    assert any(
        source["section_title"] == "Margin protection"
        for source in documents.values()
    )
    assert "TXN-000001" not in serialized
    assert "customer_type" not in serialized


def test_recommendation_evidence_and_fingerprint_are_stable() -> None:
    data = validate_sales_data(
        pd.read_csv(PROJECT_ROOT / "sample_data" / "sample_sales_data.csv")
    )

    first = build_recommendation_context(data)
    second = build_recommendation_context(data.copy())
    analytics = recommendation_analytics_evidence(first)

    assert "kpi.total_revenue" in analytics
    assert recommendation_context_fingerprint(first) == (
        recommendation_context_fingerprint(second)
    )
