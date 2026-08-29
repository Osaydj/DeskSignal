"""Whitelist dispatcher connecting Claude tool calls to safe Pandas functions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, ValidationError

from desksignal.ai.tool_models import (
    CalculateMetricInput,
    ComparePeriodsInput,
    DimensionPerformanceInput,
    SearchKnowledgeInput,
    TimeSeriesInput,
)
from desksignal.analytics.query_tools import (
    apply_analysis_filters,
    calculate_additive_metric,
    calculate_profit_margin,
    compare_periods,
    dimension_performance,
    get_time_series,
)
from desksignal.errors import ToolExecutionError, ToolInputError
from desksignal.rag.models import DocumentType
from desksignal.rag.retriever import knowledge_base_metadata, search_knowledge_base


MAX_TIME_SERIES_POINTS = 120
MAX_METADATA_LABELS = 30


@dataclass(frozen=True)
class ToolExecution:
    """A JSON-ready result plus facts Claude may cite."""

    result: dict[str, Any]
    evidence: dict[str, str]
    evidence_type: Literal["analytics", "company_document"] = "analytics"
    document_sources: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_tool_result(self) -> dict[str, Any]:
        """Return the compact payload sent back to Claude."""

        catalog: list[dict[str, Any]] = []
        for evidence_id, fact in self.evidence.items():
            item: dict[str, Any] = {
                "id": evidence_id,
                "source_type": self.evidence_type,
                "fact": fact,
            }
            if evidence_id in self.document_sources:
                item["source"] = self.document_sources[evidence_id]
            catalog.append(item)

        return {
            "result": self.result,
            "evidence_catalog": catalog,
        }


@dataclass(frozen=True)
class ToolSpec:
    """One approved tool's validation model, handler, and API description."""

    input_model: type[BaseModel]
    handler: Callable[[pd.DataFrame, BaseModel, str], ToolExecution]
    description: str


def _optional_date(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


def _clean_label(value: object) -> str:
    normalized = " ".join(str(value).replace("<", "").replace(">", "").split())
    return normalized[:120] or "Unknown"


def _round_number(value: object) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else round(numeric, 2)


def _format_metric(metric: str, value: object) -> str:
    numeric = float(value)
    if metric in {"revenue", "profit"}:
        return f"${numeric:,.2f}"
    if metric == "profit_margin":
        return f"{numeric:,.2f}%"
    return f"{int(round(numeric)):,}"


def _filtered_data(data: pd.DataFrame, args: Any) -> pd.DataFrame:
    filtered = apply_analysis_filters(
        data,
        start_date=_optional_date(args.start_date),
        end_date=_optional_date(args.end_date),
        product=args.product,
        category=args.category,
        region=args.region,
    )
    if filtered.empty:
        raise ToolExecutionError(
            "No rows match those dates and exact filters. Use a label from the "
            "current dataset metadata or remove the additional filter."
        )
    return filtered


def _scope_description(data: pd.DataFrame, args: Any) -> str:
    parts = [
        f"{data['date'].min():%Y-%m-%d} through {data['date'].max():%Y-%m-%d}"
    ]
    for field_name in ("product", "category", "region"):
        value = getattr(args, field_name, "")
        if value:
            parts.append(f"{field_name} {_clean_label(value)}")
    return ", ".join(parts)


def _calculate_revenue(
    data: pd.DataFrame,
    raw_args: BaseModel,
    prefix: str,
) -> ToolExecution:
    args = CalculateMetricInput.model_validate(raw_args)
    filtered = _filtered_data(data, args)
    revenue = float(calculate_additive_metric(filtered, "revenue"))
    evidence_id = f"{prefix}.total_revenue"
    return ToolExecution(
        result={"revenue": round(revenue, 2), "rows_analyzed": len(filtered)},
        evidence={
            evidence_id: (
                f"Revenue for {_scope_description(filtered, args)}: "
                f"{_format_metric('revenue', revenue)}."
            )
        },
    )


def _calculate_profit(
    data: pd.DataFrame,
    raw_args: BaseModel,
    prefix: str,
) -> ToolExecution:
    args = CalculateMetricInput.model_validate(raw_args)
    filtered = _filtered_data(data, args)
    profit = float(calculate_additive_metric(filtered, "profit"))
    margin = calculate_profit_margin(filtered)
    return ToolExecution(
        result={
            "profit": round(profit, 2),
            "profit_margin": round(margin, 2),
            "rows_analyzed": len(filtered),
        },
        evidence={
            f"{prefix}.total_profit": (
                f"Profit for {_scope_description(filtered, args)}: "
                f"{_format_metric('profit', profit)}."
            ),
            f"{prefix}.profit_margin": (
                f"Profit margin for {_scope_description(filtered, args)}: "
                f"{_format_metric('profit_margin', margin)}."
            ),
        },
    )


def _comparison_fact(
    *,
    metric: str,
    label: str,
    first_value: float,
    second_value: float,
    percent_change: float | None,
) -> str:
    change = second_value - first_value
    if change > 0:
        direction = "increased"
    elif change < 0:
        direction = "decreased"
    else:
        direction = "did not change"

    change_text = _format_metric(metric, abs(change))
    percent_text = (
        "percentage change is undefined because the first period was zero"
        if percent_change is None
        else f"a {abs(percent_change):,.2f}% change"
    )
    return (
        f"{label} {metric.replace('_', ' ')} {direction} from "
        f"{_format_metric(metric, first_value)} to "
        f"{_format_metric(metric, second_value)}; the absolute difference was "
        f"{change_text} and {percent_text}."
    )


def _compare_periods(
    data: pd.DataFrame,
    raw_args: BaseModel,
    prefix: str,
) -> ToolExecution:
    args = ComparePeriodsInput.model_validate(raw_args)
    calculated = compare_periods(
        data,
        period_1_start=date.fromisoformat(args.period_1_start),
        period_1_end=date.fromisoformat(args.period_1_end),
        period_2_start=date.fromisoformat(args.period_2_start),
        period_2_end=date.fromisoformat(args.period_2_end),
        metric=args.metric,
        group_by=args.group_by,
        limit=args.limit,
    )
    overall = calculated["overall"]
    first_value = float(overall["period_1_value"])
    second_value = float(overall["period_2_value"])
    percent_change = overall["percent_change"]
    evidence: dict[str, str] = {
        f"{prefix}.period_1": (
            f"{args.metric.replace('_', ' ').title()} from {args.period_1_start} "
            f"through {args.period_1_end}: "
            f"{_format_metric(args.metric, first_value)}."
        ),
        f"{prefix}.period_2": (
            f"{args.metric.replace('_', ' ').title()} from {args.period_2_start} "
            f"through {args.period_2_end}: "
            f"{_format_metric(args.metric, second_value)}."
        ),
        f"{prefix}.overall_change": _comparison_fact(
            metric=args.metric,
            label="Overall",
            first_value=first_value,
            second_value=second_value,
            percent_change=None if percent_change is None else float(percent_change),
        ),
    }

    result_rows: dict[str, list[dict[str, Any]]] = {
        "largest_increases": [],
        "largest_declines": [],
    }
    for result_name in result_rows:
        for position, row in enumerate(calculated[result_name], start=1):
            label = _clean_label(row[args.group_by])
            evidence_id = f"{prefix}.{result_name}.{position:02d}"
            row_percent = row["percent_change"]
            evidence[evidence_id] = _comparison_fact(
                metric=args.metric,
                label=f"{args.group_by.title()} {label}",
                first_value=float(row["period_1_value"]),
                second_value=float(row["period_2_value"]),
                percent_change=None if row_percent is None else float(row_percent),
            )
            result_rows[result_name].append(
                {
                    args.group_by: label,
                    "period_1_value": _round_number(row["period_1_value"]),
                    "period_2_value": _round_number(row["period_2_value"]),
                    "absolute_change": _round_number(row["absolute_change"]),
                    "percent_change": _round_number(row_percent),
                    "evidence_id": evidence_id,
                }
            )

    return ToolExecution(
        result={
            "metric": args.metric,
            "group_by": args.group_by,
            "period_1": {
                "start": args.period_1_start,
                "end": args.period_1_end,
                "value": _round_number(first_value),
            },
            "period_2": {
                "start": args.period_2_start,
                "end": args.period_2_end,
                "value": _round_number(second_value),
            },
            "absolute_change": _round_number(overall["absolute_change"]),
            "percent_change": _round_number(percent_change),
            **result_rows,
        },
        evidence=evidence,
    )


def _dimension_handler(dimension: str) -> Callable[[pd.DataFrame, BaseModel, str], ToolExecution]:
    def handler(
        data: pd.DataFrame,
        raw_args: BaseModel,
        prefix: str,
    ) -> ToolExecution:
        args = DimensionPerformanceInput.model_validate(raw_args)
        filtered = _filtered_data(data, args)
        ranked = dimension_performance(
            filtered,
            dimension=dimension,
            metric=args.metric,
            order=args.order,
            limit=args.limit,
        )
        evidence: dict[str, str] = {}
        rows: list[dict[str, Any]] = []

        for position, (_, row) in enumerate(ranked.iterrows(), start=1):
            label = _clean_label(row[dimension])
            evidence_id = f"{prefix}.rank_{position:02d}"
            evidence[evidence_id] = (
                f"{dimension.title()} {label} — revenue: "
                f"{_format_metric('revenue', row['revenue'])}; profit: "
                f"{_format_metric('profit', row['profit'])}; profit margin: "
                f"{_format_metric('profit_margin', row['profit_margin'])}; "
                f"units sold: {_format_metric('units_sold', row['units_sold'])}; "
                f"transactions: {_format_metric('transactions', row['transactions'])}."
            )
            rows.append(
                {
                    dimension: label,
                    "revenue": _round_number(row["revenue"]),
                    "profit": _round_number(row["profit"]),
                    "profit_margin": _round_number(row["profit_margin"]),
                    "units_sold": _round_number(row["units_sold"]),
                    "transactions": _round_number(row["transactions"]),
                    "evidence_id": evidence_id,
                }
            )

        return ToolExecution(
            result={
                "dimension": dimension,
                "ranked_by": args.metric,
                "order": args.order,
                "scope": _scope_description(filtered, args),
                "rows": rows,
            },
            evidence=evidence,
        )

    return handler


def _get_time_series(
    data: pd.DataFrame,
    raw_args: BaseModel,
    prefix: str,
) -> ToolExecution:
    args = TimeSeriesInput.model_validate(raw_args)
    filtered = _filtered_data(data, args)
    series = get_time_series(
        filtered,
        metric=args.metric,
        frequency=args.frequency,
    )
    if len(series) > MAX_TIME_SERIES_POINTS:
        raise ToolExecutionError(
            "That time series would return too many points. Choose a coarser "
            "frequency or a shorter date range."
        )

    evidence: dict[str, str] = {}
    points: list[dict[str, Any]] = []
    period_format = "%Y-%m-%d" if args.frequency != "month" else "%Y-%m"
    for position, (_, row) in enumerate(series.iterrows(), start=1):
        period = row["period"].strftime(period_format)
        evidence_id = f"{prefix}.point_{position:03d}"
        evidence[evidence_id] = (
            f"{args.metric.replace('_', ' ').title()} for {args.frequency} "
            f"period {period}: {_format_metric(args.metric, row['value'])}."
        )
        points.append(
            {
                "period": period,
                "value": _round_number(row["value"]),
                "evidence_id": evidence_id,
            }
        )

    return ToolExecution(
        result={
            "metric": args.metric,
            "frequency": args.frequency,
            "scope": _scope_description(filtered, args),
            "points": points,
        },
        evidence=evidence,
    )


def _search_company_knowledge(
    data: pd.DataFrame,
    raw_args: BaseModel,
    prefix: str,
) -> ToolExecution:
    del data  # Retrieval is intentionally isolated from transaction rows.
    args = SearchKnowledgeInput.model_validate(raw_args)
    selected_type: DocumentType | None = (
        None if args.document_type == "all" else args.document_type
    )
    retrieved = search_knowledge_base(
        args.query,
        top_k=args.top_k,
        document_type=selected_type,
    )
    if not retrieved:
        raise ToolExecutionError(
            "No relevant knowledge-base sections were found. Try a more specific "
            "product, policy, event, or strategy search phrase."
        )

    evidence: dict[str, str] = {}
    sources: dict[str, dict[str, Any]] = {}
    sections: list[dict[str, Any]] = []
    for position, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        evidence_id = f"{prefix}.document_{position:02d}"
        evidence[evidence_id] = chunk.text
        source = {
            "document_title": chunk.document_title,
            "section_title": chunk.section_title,
            "document_type": chunk.document_type,
            "source_path": chunk.source_path,
            "chunk_id": chunk.chunk_id,
            "relevance_score": item.score,
        }
        sources[evidence_id] = source
        sections.append(
            {
                "evidence_id": evidence_id,
                **source,
                "excerpt": chunk.text,
            }
        )

    return ToolExecution(
        result={
            "query": args.query,
            "document_type": args.document_type,
            "retrieval_method": "local TF-IDF cosine similarity",
            "retrieved_sections": sections,
        },
        evidence=evidence,
        evidence_type="company_document",
        document_sources=sources,
    )


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "calculate_revenue": ToolSpec(
        input_model=CalculateMetricInput,
        handler=_calculate_revenue,
        description=(
            "Calculate total revenue from the current dashboard data using Pandas. "
            "Use this for an exact revenue total, optionally narrowed by inclusive "
            "dates or one exact product, category, or region label. Every schema "
            "field is required; pass an empty string when a filter is not needed. "
            "The tool returns calculated JSON and evidence facts, never raw rows."
        ),
    ),
    "calculate_profit": ToolSpec(
        input_model=CalculateMetricInput,
        handler=_calculate_profit,
        description=(
            "Calculate total profit and profit margin from the current dashboard "
            "data using Pandas. Use this for an exact profit result, optionally "
            "narrowed by inclusive dates or exact product, category, and region "
            "labels. Pass an empty string for each unused filter. The result contains "
            "only aggregates and evidence facts."
        ),
    ),
    "compare_periods": ToolSpec(
        input_model=ComparePeriodsInput,
        handler=_compare_periods,
        description=(
            "Compare one metric across two explicit inclusive date ranges using "
            "Pandas. Use group_by='overall' for a direct Q1-versus-Q2 style answer, "
            "or group by product, region, or category to identify the largest "
            "positive and negative contributors. This reports association and change, "
            "not real-world causation. Limit controls each increase/decline list."
        ),
    ),
    "analyze_product_performance": ToolSpec(
        input_model=DimensionPerformanceInput,
        handler=_dimension_handler("product"),
        description=(
            "Rank products using Python-calculated revenue, profit, margin, units, "
            "or transactions. Use order='highest' for strongest products and "
            "order='lowest' for weakest products. Optional exact filters and dates "
            "can narrow the current dashboard data; pass empty strings when unused. "
            "The result contains at most the requested limit and no raw transactions."
        ),
    ),
    "analyze_region_performance": ToolSpec(
        input_model=DimensionPerformanceInput,
        handler=_dimension_handler("region"),
        description=(
            "Rank regions using Python-calculated revenue, profit, margin, units, "
            "or transactions. Use this for strongest or weakest regional performance. "
            "Dates and exact product/category/region filters narrow the current "
            "dashboard scope; pass empty strings when unused. Results are aggregate "
            "facts only and are capped by limit."
        ),
    ),
    "analyze_category_performance": ToolSpec(
        input_model=DimensionPerformanceInput,
        handler=_dimension_handler("category"),
        description=(
            "Rank categories using Python-calculated revenue, profit, margin, units, "
            "or transactions. Use order='lowest' to identify the category with the "
            "lowest margin. Dates and exact filters may narrow the dashboard data; "
            "pass empty strings when unused. Results contain aggregates and evidence, "
            "never raw CSV rows."
        ),
    ),
    "get_time_series": ToolSpec(
        input_model=TimeSeriesInput,
        handler=_get_time_series,
        description=(
            "Return a day, week, or month time series for revenue, profit, units, or "
            "transactions using Pandas. Use it to locate rises, declines, seasonality, "
            "or the period mentioned in a question. Dates and exact filters can narrow "
            "the current dashboard data; pass empty strings when unused. More than "
            "the safe point limit is rejected, so choose a coarser frequency if needed."
        ),
    ),
    "search_company_knowledge": ToolSpec(
        input_model=SearchKnowledgeInput,
        handler=_search_company_knowledge,
        description=(
            "Retrieve a small number of relevant sections from the fictional local "
            "company knowledge base. Use this for business policies, product notes, "
            "promotion schedules, inventory context, and company strategy. Provide a "
            "focused query and choose all documents or one document type. The tool "
            "returns only the top matching sections with source metadata; it never "
            "returns the complete knowledge base and never accesses arbitrary files."
        ),
    ),
}


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return strict Claude tool definitions generated from validated models."""

    definitions: list[dict[str, Any]] = []
    for name, spec in TOOL_REGISTRY.items():
        schema = anthropic.transform_schema(spec.input_model.model_json_schema())
        definitions.append(
            {
                "name": name,
                "description": spec.description,
                "strict": True,
                "input_schema": schema,
            }
        )
    return definitions


def execute_tool(
    name: str,
    tool_input: object,
    data: pd.DataFrame,
    *,
    evidence_prefix: str,
) -> ToolExecution:
    """Validate and execute one explicitly approved analytics tool."""

    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        raise ToolInputError(f"Unknown approved tool: {name}")
    if not isinstance(tool_input, dict):
        raise ToolInputError("Tool input must be a JSON object.")

    try:
        validated_input = spec.input_model.model_validate(tool_input)
    except (ValidationError, ValueError) as exc:
        raise ToolInputError(f"Invalid input for {name}: {exc}") from exc

    return spec.handler(data, validated_input, evidence_prefix)


def dataset_metadata(data: pd.DataFrame) -> dict[str, Any]:
    """Return compact labels and scope Claude can use to select tool arguments."""

    def label_summary(column: str) -> dict[str, Any]:
        labels = sorted(_clean_label(value) for value in data[column].unique())
        return {
            "total": len(labels),
            "included": labels[:MAX_METADATA_LABELS],
            "truncated": len(labels) > MAX_METADATA_LABELS,
        }

    return {
        "start_date": data["date"].min().strftime("%Y-%m-%d"),
        "end_date": data["date"].max().strftime("%Y-%m-%d"),
        "rows": len(data),
        "transactions": int(data["transaction_id"].nunique()),
        "products": label_summary("product"),
        "categories": label_summary("category"),
        "regions": label_summary("region"),
        "knowledge_base": knowledge_base_metadata(),
    }


def dataset_fingerprint(data: pd.DataFrame) -> str:
    """Create a stable local-only fingerprint for filter-aware chat history."""

    columns = [
        "transaction_id",
        "date",
        "product",
        "category",
        "region",
        "units_sold",
        "revenue",
        "profit",
    ]
    row_hashes = pd.util.hash_pandas_object(data[columns], index=False)
    return hashlib.sha256(row_hashes.values.tobytes()).hexdigest()
