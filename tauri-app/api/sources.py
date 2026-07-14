"""The source-PDF manifest written by PDF ingest."""

import json

from paths import SOURCES_MANIFEST


def load_sources() -> dict:
    """stem -> {pdf, ocr_scale}. Empty when no PDF has been ingested."""
    try:
        return json.loads(SOURCES_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}
