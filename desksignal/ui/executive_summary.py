"""Streamlit interface for the Claude-powered executive summary."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

from desksignal.ai.client import ExecutiveSummaryResult, generate_executive_summary
from desksignal.ai.context import (
    build_analysis_context,
    context_fingerprint,
    evidence_lookup,
)
from desksignal.ai.models import ExecutiveSummary, iter_summary_sections
from desksignal.errors import AIServiceError, MissingAPIKeyError


LOGGER = logging.getLogger(__name__)
SUMMARY_STATE_KEY = "ai_executive_summary"
FINGERPRINT_STATE_KEY = "ai_executive_summary_fingerprint"


def _render_summary(
    summary: ExecutiveSummary,
    context: dict[str, Any],
) -> None:
    """Render deterministic facts separately from Claude interpretations."""

    facts = evidence_lookup(context)
    for section_label, insights in iter_summary_sections(summary):
        st.subheader(section_label)
        if not insights:
            st.write("No rule-based unusual changes were identified.")
            continue

        for insight in insights:
            st.markdown(f"**{insight.title}**")
            st.markdown("**Facts calculated by Python**")
            for evidence_id in insight.evidence_ids:
                st.write(f"- {facts[evidence_id]}")
            st.markdown("**Claude interpretation**")
            st.write(insight.interpretation)


def render_ai_executive_summary(data: pd.DataFrame) -> None:
    """Render a button-driven, filter-aware executive summary section."""

    st.divider()
    st.header("AI Executive Summary")
    st.caption(
        "Python calculates the evidence first. Only aggregated results—not raw "
        "transaction rows—are sent to Claude. API usage may incur a small cost."
    )

    context = build_analysis_context(data)
    fingerprint = context_fingerprint(context)
    saved_result = st.session_state.get(SUMMARY_STATE_KEY)
    saved_fingerprint = st.session_state.get(FINGERPRINT_STATE_KEY)

    if saved_result is not None and saved_fingerprint != fingerprint:
        st.info(
            "The filters changed. Generate a new summary so its evidence matches "
            "the current dashboard."
        )

    if st.button("Generate AI Executive Summary", type="primary"):
        try:
            with st.spinner("Claude is analyzing the calculated results..."):
                result = generate_executive_summary(context)
        except MissingAPIKeyError as exc:
            st.warning(str(exc))
        except AIServiceError as exc:
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Unexpected error while generating AI summary")
            st.error(
                "An unexpected error occurred while generating the summary. "
                "Check the terminal log and try again."
            )
        else:
            st.session_state[SUMMARY_STATE_KEY] = result
            st.session_state[FINGERPRINT_STATE_KEY] = fingerprint
            saved_result = result
            saved_fingerprint = fingerprint

    if (
        isinstance(saved_result, ExecutiveSummaryResult)
        and saved_fingerprint == fingerprint
    ):
        st.caption(f"Generated with {saved_result.model}.")
        _render_summary(saved_result.summary, context)
