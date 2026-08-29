"""Application-wide configuration values."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

APP_TITLE = "InsightIQ"
APP_TAGLINE = "AI Business Analytics Assistant"
APP_ICON = "📊"
APP_PHASE = "Version 2 Preview"
CURRENCY_SYMBOL = "$"


def ai_features_enabled() -> bool:
    """Return whether paid Claude-powered features should be displayed."""

    return os.getenv("AI_FEATURES_ENABLED", "false").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
