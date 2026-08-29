"""Tests for Claude error handling and response grounding."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from desksignal.ai.client import generate_executive_summary
from desksignal.ai.models import EvidenceBasedInsight, ExecutiveSummary
from desksignal.errors import (
    AIServiceError,
    MalformedAIResponseError,
    MissingAPIKeyError,
)


class FakeMessages:
    """Record parse arguments and return or raise a prepared result."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.last_request: dict[str, Any] | None = None

    def parse(self, **kwargs: Any) -> Any:
        self.last_request = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


@pytest.fixture
def analysis_context() -> dict[str, Any]:
    return {
        "kpis": {"total_revenue": 80.0},
        "evidence_catalog": [
            {"id": "kpi.total_revenue", "fact": "Total revenue: $80.00."},
            {"id": "kpi.total_profit", "fact": "Total profit: $30.00."},
        ],
    }


def _valid_summary() -> ExecutiveSummary:
    insight = EvidenceBasedInsight(
        title="Healthy commercial result",
        evidence_ids=["kpi.total_revenue"],
        interpretation="The current result may provide a useful base for deeper review.",
    )
    return ExecutiveSummary(
        important_trends=[insight],
        strongest_areas=[insight],
        weakest_areas=[insight],
        unusual_changes=[],
        areas_to_investigate=[insight],
    )


def test_successful_response_is_structured_and_grounded(
    analysis_context: dict[str, Any],
) -> None:
    messages = FakeMessages(
        response=SimpleNamespace(
            parsed_output=_valid_summary(),
            stop_reason="end_turn",
            model="claude-sonnet-5",
        )
    )

    result = generate_executive_summary(
        analysis_context,
        api_key="test-key",
        client=FakeClient(messages),
    )

    assert result.model == "claude-sonnet-5"
    assert result.summary.important_trends[0].evidence_ids == [
        "kpi.total_revenue"
    ]
    assert messages.last_request is not None
    assert messages.last_request["output_format"] is ExecutiveSummary
    assert messages.last_request["thinking"] == {"type": "disabled"}
    assert "evidence_catalog" in messages.last_request["messages"][0]["content"]


def test_missing_api_key_has_a_clear_error(
    analysis_context: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError, match="No Anthropic API key"):
        generate_executive_summary(analysis_context, client=FakeClient(FakeMessages()))


def test_unknown_evidence_id_is_rejected(
    analysis_context: dict[str, Any],
) -> None:
    summary = _valid_summary()
    summary.important_trends[0].evidence_ids = ["made.up.statistic"]
    response = SimpleNamespace(
        parsed_output=summary,
        stop_reason="end_turn",
        model="claude-sonnet-5",
    )

    with pytest.raises(MalformedAIResponseError, match="not supplied"):
        generate_executive_summary(
            analysis_context,
            api_key="test-key",
            client=FakeClient(FakeMessages(response=response)),
        )


def test_numeric_claude_interpretation_is_rejected(
    analysis_context: dict[str, Any],
) -> None:
    summary = _valid_summary()
    summary.important_trends[0].interpretation = "Revenue may grow by 10 percent."
    response = SimpleNamespace(
        parsed_output=summary,
        stop_reason="end_turn",
        model="claude-sonnet-5",
    )

    with pytest.raises(MalformedAIResponseError, match="statistic"):
        generate_executive_summary(
            analysis_context,
            api_key="test-key",
            client=FakeClient(FakeMessages(response=response)),
        )


def test_empty_parsed_response_is_rejected(
    analysis_context: dict[str, Any],
) -> None:
    response = SimpleNamespace(
        parsed_output=None,
        stop_reason="end_turn",
        model="claude-sonnet-5",
    )

    with pytest.raises(MalformedAIResponseError, match="empty or malformed"):
        generate_executive_summary(
            analysis_context,
            api_key="test-key",
            client=FakeClient(FakeMessages(response=response)),
        )


def test_api_connection_failure_becomes_user_friendly_error(
    analysis_context: dict[str, Any],
) -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIConnectionError(request=request)

    with pytest.raises(AIServiceError, match="could not connect"):
        generate_executive_summary(
            analysis_context,
            api_key="test-key",
            client=FakeClient(FakeMessages(error=error)),
        )
