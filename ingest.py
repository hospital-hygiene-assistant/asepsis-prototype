"""
Module 1: Ingest
Copies docs from docs/ into knowledge_base/ unchanged.
Exists as an explicit pipeline step for future hooks (OCR, format normalisation, etc.).
"""

import shutil
from pathlib import Path

DOCS_DIR = Path("docs")
KB_DIR = Path("knowledge_base")


def ingest():
    KB_DIR.mkdir(exist_ok=True)
    docs = sorted(DOCS_DIR.glob("*.md"))
    if not docs:
        print(f"No markdown files found in {DOCS_DIR}/")
        return

    for src in docs:
        dst = KB_DIR / src.name
        shutil.copy2(src, dst)
        print(f"  {src.name} → {dst}")

    print(f"\nIngested {len(docs)} documents into {KB_DIR}/")


if __name__ == "__main__":
    ingest()
