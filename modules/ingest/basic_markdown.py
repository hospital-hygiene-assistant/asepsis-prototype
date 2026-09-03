"""
Module 1 — Ingest: Basic Markdown
Copies .md files from data/ to knowledge_base/ unchanged.
Simple passthrough for already well-structured markdown documents.
"""
import shutil
from pathlib import Path

from modules.ingest._manifest import Manifest, discover_files

MODULE_INFO = {
    "stage": "ingest",
    "name": "basic_markdown",
    "label": "Basic Markdown",
    "description": "Copies .md files from data/ (recursively) to knowledge_base/ unchanged. No transformation — assumes documents are already well-structured markdown. Sub-folder names become tags.",
}

DOCS_DIR = Path("data")
KB_DIR   = Path("knowledge_base")


def run(source_dir: str | None = None) -> dict:
    """Copy every .md under the source folder into knowledge_base/.

    Recursive: sub-folders are how a corpus expresses provenance ("internal"
    vs "arxiv"), and each folder name on a document's path becomes a tag.
    Documents are keyed by a path-derived id, so two files with the same name
    in different folders no longer overwrite each other.
    """
    root = Path(source_dir or DOCS_DIR)
    KB_DIR.mkdir(parents=True, exist_ok=True)
    docs = (sorted(root.glob("*/auto/*.md")) if source_dir is None
            else discover_files(root, (".md",)))
    if not docs:
        print(f"No markdown files found under {root}/")
        return {"docs": [], "warnings": []}

    manifest = Manifest(KB_DIR)
    stems = []
    for src in docs:
        entry = manifest.register(source_path=src, root=root,
                                  ingest_module=MODULE_INFO["name"])
        dst = KB_DIR / f"{entry.doc_id}.md"
        shutil.copy2(src, dst)
        tag_note = f"  [{', '.join(entry.effective_tags)}]" if entry.effective_tags else ""
        print(f"  {src.relative_to(root)} → {dst.name}{tag_note}")
        stems.append(entry.doc_id)

    print(f"\nIngested {len(docs)} documents into {KB_DIR}/")
    return {"docs": stems, "warnings": []}
