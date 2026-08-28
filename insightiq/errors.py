"""Custom exceptions used across InsightIQ."""

from __future__ import annotations

from collections.abc import Iterable


class DataValidationError(ValueError):
    """Raised when an uploaded dataset does not meet the sales-data contract."""

    def __init__(self, messages: str | Iterable[str]) -> None:
        if isinstance(messages, str):
            normalized_messages = (messages,)
        else:
            normalized_messages = tuple(messages)

        if not normalized_messages:
            normalized_messages = ("The dataset is invalid.",)

        self.messages = normalized_messages
        super().__init__(" ".join(normalized_messages))


class AIServiceError(RuntimeError):
    """Raised when an AI summary cannot be generated safely."""


class MissingAPIKeyError(AIServiceError):
    """Raised when ANTHROPIC_API_KEY is not configured."""


class MalformedAIResponseError(AIServiceError):
    """Raised when Claude's response fails structure or grounding checks."""


class ToolInputError(AIServiceError):
    """Raised when a requested analytics tool receives invalid arguments."""


class ToolExecutionError(AIServiceError):
    """Raised when an approved analytics calculation cannot be completed."""


class ToolLoopError(AIServiceError):
    """Raised when Claude does not finish within the allowed tool-use rounds."""
