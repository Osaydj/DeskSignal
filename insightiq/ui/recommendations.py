"""Streamlit interface for grounded AI management recommendations."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

from insightiq.ai.recommendation_context import (
    build_recommendation_context,
    recommendation_analytics_evidence,
    recommendation_context_fingerprint,
    recommendation_document_evidence,
)
from insightiq.ai.recommendation_models import AIRecommendations
from insightiq.ai.recommendations import (
    AIRecommendationsResult,
    generate_ai_recommendations,
)
from insightiq.errors import AIServiceError, MissingAPIKeyError


LOGGER = logging.getLogger(__name__)
RECOMMENDATIONS_STATE_KEY = "ai_management_recommendations"
FINGERPRINT_STATE_KEY = "ai_management_recommendations_fingerprint"


def _render_recommendations(
    recommendations: AIRecommendations,
    context: dict[str, Any],
) -> None:
    """Display AI reasoning beside separately resolved evidence."""

    analytics = recommendation_analytics_evidence(context)
    documents = recommendation_document_evidence(context)

    for position, item in enumerate(recommendations.recommendations, start=1):
        st.subheader(f"Recommendation {position}")
        st.markdown("**Observation**")
        st.write(item.observation)

        st.markdown("**Supporting data calculated by Python**")
        for evidence_id in item.supporting_analytics_evidence_ids:
            st.write(f"- {analytics[evidence_id]}")

        if item.supporting_document_evidence_ids:
            st.markdown("**Relevant company context**")
            for evidence_id in item.supporting_document_evidence_ids:
                source = documents[evidence_id]
                with st.expander(
                    f"{source['document_title']} — {source['section_title']}"
                ):
                    st.caption(
                        f"{source['document_type'].title()} · "
                        f"{source['source_path']}"
                    )
                    st.write(source["excerpt"])

        st.markdown("**Potential action to investigate**")
        st.write(item.suggested_action)
        st.markdown("**Reasoning**")
        st.write(item.reasoning)
        st.markdown("**Confidence and limitations**")
        st.write(f"Confidence: {item.confidence.title()}")
        st.write(item.limitations)
        st.divider()


def render_ai_recommendations(data: pd.DataFrame) -> None:
    """Render filter-aware, button-driven management recommendations."""

    st.divider()
    st.header("AI Recommendations")
    st.caption(
        "Claude proposes investigation ideas using only Python-calculated evidence "
        "and locally retrieved fictional-company sections. These are decision-support "
        "leads, not guaranteed instructions. API usage may incur a small cost."
    )

    context = build_recommendation_context(data)
    fingerprint = recommendation_context_fingerprint(context)
    saved_result = st.session_state.get(RECOMMENDATIONS_STATE_KEY)
    saved_fingerprint = st.session_state.get(FINGERPRINT_STATE_KEY)

    if saved_result is not None and saved_fingerprint != fingerprint:
        st.info(
            "The filters or company context changed. Generate new recommendations "
            "so every citation matches the current dashboard."
        )

    if st.button("Generate AI Recommendations"):
        try:
            with st.spinner(
                "Calculating evidence, retrieving context, and drafting investigation ideas..."
            ):
                result = generate_ai_recommendations(context)
        except MissingAPIKeyError as exc:
            st.warning(str(exc))
        except AIServiceError as exc:
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Unexpected error while generating AI recommendations")
            st.error(
                "An unexpected error occurred while generating recommendations. "
                "Check the terminal log and try again."
            )
        else:
            st.session_state[RECOMMENDATIONS_STATE_KEY] = result
            st.session_state[FINGERPRINT_STATE_KEY] = fingerprint
            saved_result = result
            saved_fingerprint = fingerprint

    if (
        isinstance(saved_result, AIRecommendationsResult)
        and saved_fingerprint == fingerprint
    ):
        st.caption(f"Generated with {saved_result.model}.")
        _render_recommendations(saved_result.recommendations, context)
