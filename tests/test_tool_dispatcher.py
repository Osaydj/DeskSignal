"""Tests for the analytics-tool whitelist and validation boundary."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from insightiq.ai.tool_dispatcher import (
    execute_tool,
    get_tool_definitions,
)
from insightiq.data.validation import validate_sales_data
from insightiq.errors import ToolInputError


EMPTY_FILTERS = {
    "start_date": "",
    "end_date": "",
    "product": "",
    "category": "",
    "region": "",
}


def test_all_tool_schemas_are_strict_and_fully_required() -> None:
    tools = get_tool_definitions()

    assert len(tools) == 8
    assert all(tool["strict"] is True for tool in tools)
    assert all(
        set(tool["input_schema"]["required"])
        == set(tool["input_schema"]["properties"])
        for tool in tools
    )
    assert all(
        tool["input_schema"]["additionalProperties"] is False for tool in tools
    )


def test_whitelisted_tool_returns_aggregates_and_evidence(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    execution = execute_tool(
        "analyze_region_performance",
        {
            **EMPTY_FILTERS,
            "metric": "profit",
            "order": "highest",
            "limit": 1,
        },
        data,
        evidence_prefix="call_01",
    )
    serialized = json.dumps(execution.to_tool_result())

    assert execution.result["rows"][0]["region"] == "West"
    assert "call_01.rank_01" in execution.evidence
    assert "TXN-001" not in serialized


def test_unknown_tool_and_extra_arguments_are_rejected(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    with pytest.raises(ToolInputError, match="Unknown approved tool"):
        execute_tool(
            "run_python",
            {},
            data,
            evidence_prefix="call_01",
        )

    with pytest.raises(ToolInputError, match="Invalid input"):
        execute_tool(
            "calculate_revenue",
            {**EMPTY_FILTERS, "python_code": "import os"},
            data,
            evidence_prefix="call_01",
        )


def test_knowledge_tool_returns_only_ranked_sections_with_sources(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    execution = execute_tool(
        "search_company_knowledge",
        {
            "query": "HD Webcam July promotion marketing placement change",
            "document_type": "all",
            "top_k": 2,
        },
        data,
        evidence_prefix="call_01",
    )
    serialized = json.dumps(execution.to_tool_result())

    assert execution.evidence_type == "company_document"
    assert len(execution.result["retrieved_sections"]) == 2
    assert execution.result["retrieved_sections"][0]["document_title"] == (
        "Northstar Office Co. Product Portfolio Notes"
    )
    assert set(execution.evidence) == {
        "call_01.document_01",
        "call_01.document_02",
    }
    assert "source_path" in serialized
    assert "TXN-001" not in serialized

def test_invalid_dates_and_limits_are_rejected(
    valid_sales_data: pd.DataFrame,
) -> None:
    data = validate_sales_data(valid_sales_data)

    with pytest.raises(ToolInputError, match="Invalid input"):
        execute_tool(
            "get_time_series",
            {
                **EMPTY_FILTERS,
                "start_date": "not-a-date",
                "metric": "revenue",
                "frequency": "month",
            },
            data,
            evidence_prefix="call_01",
        )

    with pytest.raises(ToolInputError, match="Invalid input"):
        execute_tool(
            "analyze_product_performance",
            {
                **EMPTY_FILTERS,
                "metric": "revenue",
                "order": "highest",
                "limit": 100,
            },
            data,
            evidence_prefix="call_01",
        )
