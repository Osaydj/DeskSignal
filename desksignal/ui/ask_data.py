"""Streamlit chat interface for the Ask Your Data tool-use feature."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import pandas as pd
import streamlit as st

from desksignal.ai.ask_data import ask_your_data
from desksignal.ai.tool_dispatcher import dataset_fingerprint
from desksignal.errors import AIServiceError, MissingAPIKeyError, ToolInputError


LOGGER = logging.getLogger(__name__)
HISTORY_STATE_KEY = "ask_your_data_history"
FINGERPRINT_STATE_KEY = "ask_your_data_fingerprint"
MAX_STORED_MESSAGES = 30


def _render_assistant_message(message: dict[str, Any]) -> None:
    """Display calculated and retrieved evidence as separate source groups."""

    analytics_facts = message.get("analytics_facts", message.get("facts", []))
    if analytics_facts:
        st.markdown("**Analytics evidence (calculated by Python)**")
        for fact in analytics_facts:
            st.write(f"- {fact}")

    document_evidence = message.get("document_evidence", [])
    if document_evidence:
        st.markdown("**Company document evidence (retrieved sections)**")
        for item in document_evidence:
            st.markdown(
                f"**{item['document_title']} — {item['section_title']}**"
            )
            st.caption(
                f"{item['document_type'].title()} · {item['source_path']}"
            )
            st.info(item["excerpt"])

    st.markdown("**Claude explanation**")
    st.write(message["content"])

    limitations = message.get("limitations", [])
    if limitations:
        st.markdown("**Limitations**")
        for limitation in limitations:
            st.write(f"- {limitation}")

    tools_used = message.get("tools_used", [])
    if tools_used:
        with st.expander("Approved tools used"):
            for tool_name in tools_used:
                st.code(tool_name, language=None)


def _render_history(history: list[dict[str, Any]]) -> None:
    for message in history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant_message(message)
            else:
                st.write(message["content"])


def _history_for_api(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Keep only safe text fields when sending short conversation history."""

    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in history
        if message.get("role") in {"user", "assistant"}
    ]


def render_ask_your_data(data: pd.DataFrame) -> None:
    """Render filter-aware conversation history and a natural-language input."""

    st.divider()
    st.header("Ask Your Data")
    st.caption(
        "Claude chooses from approved read-only analytics and document-retrieval "
        "tools. Pandas performs every calculation, relevant knowledge-base sections "
        "are retrieved locally, and raw CSV rows are never sent to Claude."
    )

    current_fingerprint = dataset_fingerprint(data)
    previous_fingerprint = st.session_state.get(FINGERPRINT_STATE_KEY)
    filters_changed = (
        previous_fingerprint is not None
        and previous_fingerprint != current_fingerprint
    )
    if filters_changed:
        st.session_state[HISTORY_STATE_KEY] = []
        st.info(
            "Dashboard filters changed, so the previous data conversation was cleared."
        )
    st.session_state[FINGERPRINT_STATE_KEY] = current_fingerprint

    history: list[dict[str, Any]] = st.session_state.setdefault(
        HISTORY_STATE_KEY,
        [],
    )

    if history and st.button("Clear Ask Your Data conversation"):
        st.session_state[HISTORY_STATE_KEY] = []
        st.rerun()

    _render_history(history)
    question = st.chat_input(
        "Ask about revenue, products, regions, or relevant company documents"
    )
    if not question:
        return

    prior_history = _history_for_api(history)
    user_message = {"role": "user", "content": question}
    history.append(user_message)
    with st.chat_message("user"):
        st.write(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Calculating data and retrieving relevant documents..."):
                result = ask_your_data(
                    data,
                    question,
                    history=prior_history,
                )
            assistant_message = {
                "role": "assistant",
                "content": result.answer,
                "analytics_facts": list(result.analytics_facts),
                "document_evidence": [
                    asdict(item) for item in result.document_evidence
                ],
                "limitations": list(result.limitations),
                "tools_used": list(result.tools_used),
                "model": result.model,
            }
            _render_assistant_message(assistant_message)
    except MissingAPIKeyError as exc:
        if history and history[-1] is user_message:
            history.pop()
        st.warning(str(exc))
    except (ToolInputError, AIServiceError) as exc:
        if history and history[-1] is user_message:
            history.pop()
        st.error(str(exc))
    except Exception:
        if history and history[-1] is user_message:
            history.pop()
        LOGGER.exception("Unexpected Ask Your Data error")
        st.error(
            "An unexpected error occurred while answering the question. "
            "Check the terminal log and try again."
        )
    else:
        history.append(assistant_message)
        if len(history) > MAX_STORED_MESSAGES:
            del history[:-MAX_STORED_MESSAGES]
