"""Grounded Claude integration for DeskSignal."""

from desksignal.ai.ask_data import AskDataResult, ask_your_data
from desksignal.ai.client import ExecutiveSummaryResult, generate_executive_summary
from desksignal.ai.context import build_analysis_context, context_fingerprint

__all__ = [
    "AskDataResult",
    "ExecutiveSummaryResult",
    "ask_your_data",
    "build_analysis_context",
    "context_fingerprint",
    "generate_executive_summary",
]
