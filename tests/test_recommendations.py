"""Tests for structured, evidence-grounded management recommendations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from insightiq.ai.recommendation_models import (
    AIRecommendations,
    ManagementRecommendation,
)
from insightiq.ai.recommendations import generate_ai_recommendations
from insightiq.errors import MalformedAIResponseError


class FakeMessages:
    """Return a prepared parsed response and record request arguments."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.last_request: dict[str, Any] | None = None

    def parse(self, **kwargs: Any) -> Any:
        self.last_request = kwargs
        return self.response


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.messages = FakeMessages(response)


@pytest.fixture
def recommendation_context() -> dict[str, Any]:
    return {
        "analytics": {
            "evidence_catalog": [
                {
                    "id": "product.top.revenue",
                    "fact": "Top product revenue: $500.00.",
                },
                {
                    "id": "category.low.margin",
                    "fact": "Lowest category profit margin: 20.00%.",
                },
                {
                    "id": "region.low.profit",
                    "fact": "Lowest region profit: $40.00.",
                },
            ]
        },
        "company_context": {
            "retrieved_sections": [
                {
                    "evidence_id": "document.01",
                    "document_title": "Fictional Policy",
                    "section_title": "Margin review",
                    "document_type": "policy",
                    "source_path": "policies/fictional.md",
                    "excerpt": "Managers should review margin before testing discounts.",
                    "relevance_score": 0.5,
                    "retrieval_query": "margin discount policy",
                }
            ]
        },
    }


def _valid_recommendations() -> AIRecommendations:
    return AIRecommendations(
        recommendations=[
            ManagementRecommendation(
                observation="The leading product is an important performance area.",
                supporting_analytics_evidence_ids=["product.top.revenue"],
                supporting_document_evidence_ids=[],
                suggested_action=(
                    "Review which repeatable factors may support the product result."
                ),
                reasoning=(
                    "A focused review could identify factors worth testing elsewhere."
                ),
                confidence="medium",
                limitations=(
                    "The available evidence does not establish why the product leads."
                ),
            ),
            ManagementRecommendation(
                observation="The weakest category margin deserves a controlled review.",
                supporting_analytics_evidence_ids=["category.low.margin"],
                supporting_document_evidence_ids=["document.01"],
                suggested_action=(
                    "Compare discount practices with the documented margin review policy."
                ),
                reasoning=(
                    "The policy provides relevant context for investigating margin pressure."
                ),
                confidence="high",
                limitations=(
                    "The evidence cannot show whether discounts created the margin result."
                ),
            ),
            ManagementRecommendation(
                observation="The weakest regional profit is worth investigating.",
                supporting_analytics_evidence_ids=["region.low.profit"],
                supporting_document_evidence_ids=[],
                suggested_action=(
                    "Review the regional product mix before considering a small pilot."
                ),
                reasoning=(
                    "A comparison may reveal measured contributors without assuming a cause."
                ),
                confidence="low",
                limitations=(
                    "Regional cost drivers and customer conditions are not available."
                ),
            ),
        ]
    )


def _response(recommendations: AIRecommendations) -> Any:
    return SimpleNamespace(
        parsed_output=recommendations,
        stop_reason="end_turn",
        model="claude-sonnet-5",
    )


def test_recommendations_are_structured_and_grounded(
    recommendation_context: dict[str, Any],
) -> None:
    client = FakeClient(_response(_valid_recommendations()))

    result = generate_ai_recommendations(
        recommendation_context,
        api_key="test-key",
        client=client,
    )

    assert len(result.recommendations.recommendations) == 3
    assert result.model == "claude-sonnet-5"
    assert client.messages.last_request is not None
    assert client.messages.last_request["output_format"] is AIRecommendations
    assert "recommendation_context" in (
        client.messages.last_request["messages"][0]["content"]
    )


def test_unknown_analytics_evidence_is_rejected(
    recommendation_context: dict[str, Any],
) -> None:
    recommendations = _valid_recommendations()
    recommendations.recommendations[0].supporting_analytics_evidence_ids = [
        "invented.analytics"
    ]

    with pytest.raises(MalformedAIResponseError, match="analytics evidence"):
        generate_ai_recommendations(
            recommendation_context,
            api_key="test-key",
            client=FakeClient(_response(recommendations)),
        )


def test_unknown_document_evidence_is_rejected(
    recommendation_context: dict[str, Any],
) -> None:
    recommendations = _valid_recommendations()
    recommendations.recommendations[1].supporting_document_evidence_ids = [
        "document.invented"
    ]

    with pytest.raises(MalformedAIResponseError, match="not retrieved"):
        generate_ai_recommendations(
            recommendation_context,
            api_key="test-key",
            client=FakeClient(_response(recommendations)),
        )


def test_new_number_in_claude_prose_is_rejected(
    recommendation_context: dict[str, Any],
) -> None:
    recommendations = _valid_recommendations()
    recommendations.recommendations[0].reasoning = (
        "This action could increase revenue by 12 percent."
    )

    with pytest.raises(MalformedAIResponseError, match="number"):
        generate_ai_recommendations(
            recommendation_context,
            api_key="test-key",
            client=FakeClient(_response(recommendations)),
        )


def test_retrieved_context_cannot_be_silently_ignored(
    recommendation_context: dict[str, Any],
) -> None:
    recommendations = _valid_recommendations()
    for item in recommendations.recommendations:
        item.supporting_document_evidence_ids = []

    with pytest.raises(MalformedAIResponseError, match="ignored"):
        generate_ai_recommendations(
            recommendation_context,
            api_key="test-key",
            client=FakeClient(_response(recommendations)),
        )
