"""Strict request and response models for Ask Your Data."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DATE_DESCRIPTION = "Inclusive YYYY-MM-DD date, or an empty string for no date limit."
FILTER_DESCRIPTION = "Exact label from the dataset, or an empty string for no filter."


class StrictToolInput(BaseModel):
    """Base model that rejects every undeclared tool argument."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FilteredToolInput(StrictToolInput):
    """Common required fields used by filter-aware calculation tools."""

    start_date: str = Field(description=DATE_DESCRIPTION)
    end_date: str = Field(description=DATE_DESCRIPTION)
    product: str = Field(max_length=120, description=FILTER_DESCRIPTION)
    category: str = Field(max_length=120, description=FILTER_DESCRIPTION)
    region: str = Field(max_length=120, description=FILTER_DESCRIPTION)

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_string(cls, value: str) -> str:
        if value:
            date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def validate_date_order(self) -> "FilteredToolInput":
        if self.start_date and self.end_date:
            if date.fromisoformat(self.start_date) > date.fromisoformat(self.end_date):
                raise ValueError("start_date cannot be after end_date")
        return self


class CalculateMetricInput(FilteredToolInput):
    """Validated inputs for calculate_revenue and calculate_profit."""


class ComparePeriodsInput(StrictToolInput):
    """Validated inputs for comparing two time periods."""

    period_1_start: str = Field(description="First period start in YYYY-MM-DD format.")
    period_1_end: str = Field(description="First period end in YYYY-MM-DD format.")
    period_2_start: str = Field(description="Second period start in YYYY-MM-DD format.")
    period_2_end: str = Field(description="Second period end in YYYY-MM-DD format.")
    metric: Literal["revenue", "profit", "units_sold", "transactions"]
    group_by: Literal["overall", "product", "region", "category"]
    limit: int = Field(ge=1, le=10)

    @field_validator(
        "period_1_start",
        "period_1_end",
        "period_2_start",
        "period_2_end",
    )
    @classmethod
    def validate_required_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def validate_period_order(self) -> "ComparePeriodsInput":
        if date.fromisoformat(self.period_1_start) > date.fromisoformat(
            self.period_1_end
        ):
            raise ValueError("period_1_start cannot be after period_1_end")
        if date.fromisoformat(self.period_2_start) > date.fromisoformat(
            self.period_2_end
        ):
            raise ValueError("period_2_start cannot be after period_2_end")
        return self


class DimensionPerformanceInput(FilteredToolInput):
    """Validated inputs for product, region, and category rankings."""

    metric: Literal[
        "revenue",
        "profit",
        "profit_margin",
        "units_sold",
        "transactions",
    ]
    order: Literal["highest", "lowest"]
    limit: int = Field(ge=1, le=10)


class TimeSeriesInput(FilteredToolInput):
    """Validated inputs for a time-series calculation."""

    metric: Literal["revenue", "profit", "units_sold", "transactions"]
    frequency: Literal["day", "week", "month"]


class SearchKnowledgeInput(StrictToolInput):
    """Validated inputs for local knowledge-base retrieval."""

    query: str = Field(
        min_length=3,
        max_length=300,
        description=(
            "Focused search phrase describing the policy, product, event, or "
            "strategy information needed from company documents."
        ),
    )
    document_type: Literal["all", "policy", "product", "strategy"] = Field(
        description="Document type to search, or 'all' to search every type."
    )
    top_k: int = Field(
        ge=1,
        le=5,
        description="Maximum number of relevant document sections to retrieve.",
    )


class AskDataAnswer(BaseModel):
    """Claude's final narrative linked to Python-generated evidence."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=10,
        max_length=700,
        description=(
            "Concise explanation based only on cited evidence. Do not calculate "
            "or introduce any statistic not present in the cited facts."
        ),
    )
    analytics_evidence_ids: list[str] = Field(
        max_length=10,
        description=(
            "Analytics evidence IDs copied exactly from successful Pandas tool results."
        ),
    )
    document_evidence_ids: list[str] = Field(
        max_length=10,
        description=(
            "Document evidence IDs copied exactly from retrieved knowledge sections."
        ),
    )
    limitations: list[str] = Field(
        max_length=3,
        description=(
            "Short cautions about what the calculated data cannot establish."
        ),
    )

    @model_validator(mode="after")
    def require_some_evidence(self) -> "AskDataAnswer":
        if not self.analytics_evidence_ids and not self.document_evidence_ids:
            raise ValueError("At least one analytics or document evidence ID is required")
        return self
