"""Where the project keeps its data.

Anchored to the repo rather than the working directory: the server, the pipeline
CLI and the Tauri launcher all start from different places, and a relative path
resolves to an empty corpus instead of failing.

One definition per location. The source-candidate manifest is written by ingest
adapters and consumed only when the Expected library is published.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DOCS_DIR = ROOT / "docs"
"""Source documents, as supplied. Ingest reads from here and never writes."""

KB_DIR = ROOT / "knowledge_base"
"""Ingest staging for canonical Markdown before immutable publication."""

LIBRARY_DIR = ROOT / "library"
"""Complete immutable Expected library generations and source objects."""

ASSETS_DIR = KB_DIR / "assets"
"""Ingest staging for figure/table crops before immutable publication."""

SOURCES_MANIFEST = KB_DIR / ".sources.json"
"""Versioned source candidates written by ingest and consumed at publication."""
