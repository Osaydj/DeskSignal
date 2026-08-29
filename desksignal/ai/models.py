"""Validated response models for Claude's executive summary."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, Field


class EvidenceBasedInsight(BaseModel):
    """One interpretation linked to deterministic evidence calculated by Python."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=3,
        max_length=80,
        description="A short qualitative heading containing no numbers.",
    )
    evidence_ids: list[str] = Field(
        min_length=1,
        max_length=3,
        description="IDs copied exactly from the supplied evidence catalog.",
    )
    interpretation: str = Field(
        min_length=10,
        max_length=320,
        description=(
            "A cautious business interpretation containing no new facts or numbers."
        ),
    )


class ExecutiveSummary(BaseModel):
    """Structured, evidence-linked business analysis returned by Claude."""

    model_config = ConfigDict(extra="forbid")

    important_trends: list[EvidenceBasedInsight] = Field(
        min_length=1,
        max_length=2,
    )
    strongest_areas: list[EvidenceBasedInsight] = Field(
        min_length=1,
        max_length=2,
    )
    weakest_areas: list[EvidenceBasedInsight] = Field(
        min_length=1,
        max_length=2,
    )
    unusual_changes: list[EvidenceBasedInsight] = Field(max_length=2)
    areas_to_investigate: list[EvidenceBasedInsight] = Field(
        min_length=1,
        max_length=2,
    )


SUMMARY_SECTION_LABELS = (
    ("Important trends", "important_trends"),
    ("Strongest areas", "strongest_areas"),
    ("Weakest areas", "weakest_areas"),
    ("Unusual changes", "unusual_changes"),
    ("Areas worth investigating", "areas_to_investigate"),
)


def iter_summary_sections(
    summary: ExecutiveSummary,
) -> Iterator[tuple[str, list[EvidenceBasedInsight]]]:
    """Yield display labels and their corresponding summary items."""

    for label, field_name in SUMMARY_SECTION_LABELS:
        yield label, getattr(summary, field_name)
