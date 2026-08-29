"""Build the calculated and retrieved evidence used for recommendations."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from desksignal.ai.context import build_analysis_context, evidence_lookup
from desksignal.rag.retriever import search_knowledge_base


MAX_RETRIEVED_SECTIONS = 5
SECTIONS_PER_QUERY = 2
MIN_RELEVANCE_SCORE = 0.10


def _retrieval_queries(analytics: dict[str, Any]) -> list[str]:
    """Create focused searches from deterministic performance results."""

    products = analytics["product_performance"]["records"]
    categories = analytics["category_performance"]["records"]
    regions = analytics["regional_performance"]["records"]
    unusual = analytics["unusual_changes"]["records"]
    queries: list[str] = []

    if products:
        queries.append(
            f"{products[0]['product']} product promotion strategy inventory"
        )
        if len(products) > 1:
            queries.append(
                f"{products[-1]['product']} product promotion strategy inventory"
            )

    if categories:
        lowest_margin_category = min(
            categories,
            key=lambda item: item["profit_margin_pct"],
        )
        queries.append(
            f"{lowest_margin_category['category']} margin discount policy"
        )

    if regions:
        lowest_profit_region = min(regions, key=lambda item: item["profit"])
        queries.append(
            f"{lowest_profit_region['region']} region allocation inventory strategy"
        )

    if unusual:
        largest_change = max(
            unusual,
            key=lambda item: abs(item["change_pct"]),
        )
        queries.append(
            f"{largest_change['from_month']} {largest_change['to_month']} "
            f"{largest_change['metric']} change promotion strategy"
        )

    # Preserve order while preventing duplicate searches.
    return list(dict.fromkeys(queries))


def _retrieve_company_sections(
    analytics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Retrieve a small, diverse set of sections relevant to calculated results."""

    selected: list[dict[str, Any]] = []
    selected_chunk_ids: set[str] = set()

    for query in _retrieval_queries(analytics):
        results = search_knowledge_base(query, top_k=SECTIONS_PER_QUERY)
        for result in results:
            chunk = result.chunk
            if result.score < MIN_RELEVANCE_SCORE:
                continue
            if chunk.chunk_id in selected_chunk_ids:
                continue

            evidence_id = f"document.{len(selected) + 1:02d}"
            selected.append(
                {
                    "evidence_id": evidence_id,
                    "document_title": chunk.document_title,
                    "section_title": chunk.section_title,
                    "document_type": chunk.document_type,
                    "source_path": chunk.source_path,
                    "excerpt": chunk.text,
                    "relevance_score": result.score,
                    "retrieval_query": query,
                }
            )
            selected_chunk_ids.add(chunk.chunk_id)
            break

        if len(selected) >= MAX_RETRIEVED_SECTIONS:
            break

    return selected


def build_recommendation_context(data: pd.DataFrame) -> dict[str, Any]:
    """Return aggregates plus only relevant fictional-company document sections."""

    analytics = build_analysis_context(data)
    sections = _retrieve_company_sections(analytics)
    return {
        "analytics": analytics,
        "company_context": {
            "retrieval_method": "local TF-IDF cosine similarity",
            "retrieved_section_count": len(sections),
            "retrieved_sections": sections,
            "retrieval_note": (
                "Sections are investigation context. Matching text and timing do "
                "not prove that a document event caused an analytics result."
            ),
        },
    }


def recommendation_analytics_evidence(
    context: dict[str, Any],
) -> dict[str, str]:
    """Return the Python-calculated evidence available to recommendations."""

    analytics = context.get("analytics")
    if not isinstance(analytics, dict):
        raise ValueError("Recommendation context is missing analytics evidence.")
    return evidence_lookup(analytics)


def recommendation_document_evidence(
    context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return retrieved document sections keyed by their evidence IDs."""

    company_context = context.get("company_context", {})
    sections = company_context.get("retrieved_sections", [])
    lookup: dict[str, dict[str, Any]] = {}
    for section in sections:
        evidence_id = str(section["evidence_id"])
        if evidence_id in lookup:
            raise ValueError(f"Duplicate document evidence ID: {evidence_id}")
        lookup[evidence_id] = section
    return lookup


def recommendation_context_fingerprint(context: dict[str, Any]) -> str:
    """Create a stable identifier for filter- and document-aware UI state."""

    serialized = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
