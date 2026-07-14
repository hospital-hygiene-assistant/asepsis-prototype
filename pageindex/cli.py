"""Building the index directly. `pipeline.py index` is the usual route."""

import argparse
import sys

from paths import KB_DIR

from .retrieve import build_index
from .settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PageIndex for knowledge_base documents.")
    parser.add_argument("--doc", help="Single doc name (stem, no .md). Default: all.")
    parser.add_argument("--model", default=settings.model,
                        help=f"Ollama model (default: {settings.model})")
    args = parser.parse_args()
    settings.model = args.model

    if args.doc:
        build_index(args.doc)
        return

    docs = sorted(KB_DIR.glob("*.md"))
    if not docs:
        print(f"No documents in {KB_DIR}/. Run 'pipeline.py ingest' first.")
        sys.exit(1)
    print(f"Building index for {len(docs)} documents...")
    for path in docs:
        build_index(path.stem)
    print("\nDone.")


if __name__ == "__main__":
    main()
