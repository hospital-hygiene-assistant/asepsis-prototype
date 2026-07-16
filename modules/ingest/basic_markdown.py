"""
Module 1 — Ingest: Basic Markdown
Copies .md files from docs/ to knowledge_base/ unchanged.
Simple passthrough for already well-structured markdown documents.
"""
import shutil

MODULE_INFO = {
    "stage": "ingest",
    "name": "basic_markdown",
    "label": "Basic Markdown",
    "description": "Copies .md files from docs/ to knowledge_base/ unchanged. No transformation — assumes documents are already well-structured markdown.",
}

from paths import DOCS_DIR, KB_DIR, SOURCES_MANIFEST
from pageindex.library import read_source_candidates, write_source_candidates


def run() -> None:
    KB_DIR.mkdir(exist_ok=True)
    docs = sorted(DOCS_DIR.glob("*.md"))
    if not docs:
        print(f"No markdown files found in {DOCS_DIR}/")
        return
    sources = read_source_candidates(SOURCES_MANIFEST)
    for src in docs:
        dst = KB_DIR / src.name
        shutil.copy2(src, dst)
        sources.pop(src.stem, None)
        print(f"  {src.name} → {dst}")
    write_source_candidates(SOURCES_MANIFEST, sources)
    print(f"\nIngested {len(docs)} documents into {KB_DIR}/")
