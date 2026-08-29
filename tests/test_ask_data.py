"""Tests for the bounded Claude tool-use loop."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from desksignal.ai.ask_data import ask_your_data
from desksignal.data.validation import validate_sales_data
from desksignal.errors import MalformedAIResponseError


class FakeMessages:
    """Return prepared Claude responses and record every API request."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(copy.deepcopy(kwargs))
        return next(self.responses)


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = FakeMessages(responses)


def _tool_response() -> Any:
    return SimpleNamespace(
        stop_reason="tool_use",
        model="claude-sonnet-5",
        content=[
            SimpleNamespace(
                type="tool_use",
                id="toolu_region",
                name="analyze_region_performance",
                input={
                    "start_date": "",
                    "end_date": "",
                    "product": "",
                    "category": "",
                    "region": "",
                    "metric": "profit",
                    "order": "highest",
                    "limit": 1,
                },
            )
        ],
    )


def _final_response(
    *,
    answer: str,
    analytics_evidence_ids: tuple[str, ...] = (),
    document_evidence_ids: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
) -> Any:
    payload = {
        "answer": answer,
        "analytics_evidence_ids": list(analytics_evidence_ids),
        "document_evidence_ids": list(document_evidence_ids),
        "limitations": list(limitations),
    }
    return SimpleNamespace(
        stop_reason="end_turn",
        model="claude-sonnet-5",
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
    )


def test_tool_loop_sends_results_back_and_returns_grounded_answer(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)
    client = FakeClient(
        [
            _tool_response(),
            _final_response(
                answer="West generated the most profit at $22.",
                analytics_evidence_ids=("call_01.rank_01",),
            ),
        ]
    )

    result = ask_your_data(
        data,
        "Which region generated the most profit?",
        api_key="test-key",
        client=client,
    )

    assert result.answer == "West generated the most profit at $22."
    assert result.analytics_facts[0].startswith("Region West")
    assert result.document_evidence == ()
    assert result.tools_used == ("analyze_region_performance",)
    assert len(client.messages.requests) == 2

    first_request = client.messages.requests[0]
    assert first_request["tool_choice"]["type"] == "any"
    assert all(tool["strict"] for tool in first_request["tools"])
    assert "TXN-001" not in json.dumps(first_request["messages"])

    second_request = client.messages.requests[1]
    tool_result = second_request["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_region"
    assert "call_01.rank_01" in tool_result["content"]


def test_unknown_final_evidence_is_rejected(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)
    client = FakeClient(
        [
            _tool_response(),
            _final_response(
                answer="West generated the most profit.",
                analytics_evidence_ids=("invented.fact",),
            ),
        ]
    )

    with pytest.raises(MalformedAIResponseError, match="not returned"):
        ask_your_data(
            data,
            "Which region generated the most profit?",
            api_key="test-key",
            client=client,
        )


def test_number_absent_from_cited_fact_is_rejected(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)
    client = FakeClient(
        [
            _tool_response(),
            _final_response(
                answer="West generated $999 in profit.",
                analytics_evidence_ids=("call_01.rank_01",),
            ),
        ]
    )

    with pytest.raises(MalformedAIResponseError, match="number"):
        ask_your_data(
            data,
            "Which region generated the most profit?",
            api_key="test-key",
            client=client,
        )


def test_number_with_wrong_unit_is_rejected(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)
    client = FakeClient(
        [
            _tool_response(),
            _final_response(
                answer="West generated a 22% profit result.",
                analytics_evidence_ids=("call_01.rank_01",),
            ),
        ]
    )

    with pytest.raises(MalformedAIResponseError, match="number"):
        ask_your_data(
            data,
            "Which region generated the most profit?",
            api_key="test-key",
            client=client,
        )


def test_hybrid_question_uses_analytics_and_retrieved_document_sections() -> None:
    data = validate_sales_data(pd.read_csv("sample_data/sample_sales_data.csv"))
    analytics_response = SimpleNamespace(
        stop_reason="tool_use",
        model="claude-sonnet-5",
        content=[
            SimpleNamespace(
                type="tool_use",
                id="toolu_series",
                name="get_time_series",
                input={
                    "start_date": "2025-06-01",
                    "end_date": "2025-07-31",
                    "product": "HD Webcam",
                    "category": "",
                    "region": "",
                    "metric": "revenue",
                    "frequency": "month",
                },
            )
        ],
    )
    retrieval_response = SimpleNamespace(
        stop_reason="tool_use",
        model="claude-sonnet-5",
        content=[
            SimpleNamespace(
                type="tool_use",
                id="toolu_knowledge",
                name="search_company_knowledge",
                input={
                    "query": "HD Webcam July promotion marketing placement change",
                    "document_type": "all",
                    "top_k": 2,
                },
            )
        ],
    )
    final_response = _final_response(
        answer=(
            "Analytics data shows a decline. Company documents describe a promotion "
            "ending and a placement change that may be relevant."
        ),
        analytics_evidence_ids=(
            "call_01.point_001",
            "call_01.point_002",
        ),
        document_evidence_ids=("call_02.document_01",),
        limitations=(
            "The timing is an association and does not prove the document event caused the decline.",
        ),
    )
    client = FakeClient(
        [analytics_response, retrieval_response, final_response]
    )

    result = ask_your_data(
        data,
        "HD Webcam sales dropped in July. Do company documents help explain it?",
        api_key="test-key",
        client=client,
    )

    assert result.tools_used == ("get_time_series", "search_company_knowledge")
    assert len(result.analytics_facts) == 2
    assert len(result.document_evidence) == 1
    assert result.document_evidence[0].document_title == (
        "Northstar Office Co. Product Portfolio Notes"
    )
    assert result.document_evidence[0].source_path.startswith("products/")
    retrieval_request = client.messages.requests[2]
    retrieval_result = retrieval_request["messages"][-1]["content"][0]
    returned_payload = json.loads(retrieval_result["content"])
    assert len(returned_payload["result"]["retrieved_sections"]) == 2
