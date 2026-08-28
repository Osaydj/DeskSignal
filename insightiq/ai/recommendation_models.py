"""Validated structured models for management recommendations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ManagementRecommendation(BaseModel):
    """One cautious potential action linked to calculated evidence."""

    model_config = ConfigDict(extra="forbid")

    observation: str = Field(
        min_length=10,
        max_length=220,
        description=(
            "A concise observation with no numbers. It must be supported by the "
            "cited analytics evidence rather than external knowledge."
        ),
    )
    supporting_analytics_evidence_ids: list[str] = Field(
        min_length=1,
        max_length=4,
        description=(
            "IDs copied exactly from the Python-calculated analytics evidence catalog."
        ),
    )
    supporting_document_evidence_ids: list[str] = Field(
        max_length=2,
        description=(
            "IDs copied exactly from retrieved company sections, or an empty list "
            "when no retrieved section is relevant."
        ),
    )
    suggested_action: str = Field(
        min_length=10,
        max_length=260,
        description=(
            "A reversible investigation, review, comparison, test, or pilot—not a "
            "guaranteed instruction. Include no numbers."
        ),
    )
    reasoning: str = Field(
        min_length=10,
        max_length=320,
        description=(
            "Cautious reasoning connecting the cited evidence to the suggested "
            "investigation, with no invented facts or numbers."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in the recommendation as an investigation lead."
    )
    limitations: str = Field(
        min_length=10,
        max_length=260,
        description=(
            "A specific uncertainty, missing fact, or causation limitation. Include "
            "no numbers."
        ),
    )


class AIRecommendations(BaseModel):
    """Three to five evidence-linked management investigation ideas."""

    model_config = ConfigDict(extra="forbid")

    recommendations: list[ManagementRecommendation] = Field(
        min_length=3,
        max_length=5,
    )
