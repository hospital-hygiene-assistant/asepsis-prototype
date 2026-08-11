"""
Module 1: Ingest — standalone entry point.

Kept because run-tauri.py and the docs invoke it directly, but the logic now
lives in modules/ingest/basic_markdown.py so there is exactly one ingest
implementation. Running this used to copy docs/*.md by filename, without
recursing into sub-folders and without writing a manifest — a library built
that way would have no document ids and no tags, and would behave differently
from one built through the app.

Usage:
    python ingest.py                # ingest docs/ recursively
    python ingest.py <folder>       # ingest another folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.ingest.basic_markdown import DOCS_DIR, KB_DIR, run


def ingest(source_dir: str | None = None) -> dict:
    return run(source_dir)


if __name__ == "__main__":
    ingest(sys.argv[1] if len(sys.argv) > 1 else None)
