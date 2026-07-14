"""Where the project keeps its data.

Anchored to the repo rather than the working directory: the server, the pipeline
CLI and the Tauri launcher all start from different places, and a relative path
resolves to an empty corpus instead of failing.

One definition per location. The source manifest in particular is written by the
PDF ingest module and read by the API, so the two cannot be allowed to drift.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DOCS_DIR = ROOT / "docs"
"""Source documents, as supplied. Ingest reads from here and never writes."""

KB_DIR = ROOT / "knowledge_base"
"""Ingested markdown. The corpus the index is built from and answers cite."""

INDEX_DIR = ROOT / "index"
"""One heading tree per document, as JSON."""

ASSETS_DIR = KB_DIR / "assets"
"""Figure and table crops lifted out during PDF ingest, served at /assets."""

SOURCES_MANIFEST = KB_DIR / ".sources.json"
"""stem -> {pdf, ocr_scale}. Written by PDF ingest, read to render source pages."""
