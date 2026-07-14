"""
Unified pipeline CLI with per-stage module selection.

Commands:
  python3 pipeline.py list
      List all available modules for each stage.

  python3 pipeline.py ingest [--module basic_markdown]
      Run the ingest stage.

  python3 pipeline.py index [--module pageindex_custom] [--doc NAME]
      Build the index (all docs, or --doc for one).

  python3 pipeline.py query [--index-module pageindex_custom]
                            [--query-module ollama_synthesis]
                            "your question"
      Run end-to-end retrieval + synthesis.

  python3 pipeline.py run   [--ingest basic_markdown]
                            [--index  pageindex_custom]
                            [--query  ollama_synthesis]
                            "your question"
      Full pipeline: ingest → index → query.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.registry import discover, load, defaults
from paths import INDEX_DIR, KB_DIR


# ── helpers ──────────────────────────────────────────────────

def _print_modules(registry: dict) -> None:
    labels = {"ingest": "Module 1 — Ingest", "index": "Module 2 — Index", "query": "Module 3 — Query"}
    for stage, mods in registry.items():
        print(f"\n{labels.get(stage, stage)}:")
        if not mods:
            print("  (none found)")
            continue
        for name, info in mods.items():
            marker = " *" if name == defaults()[stage] else "  "
            print(f"{marker} {name}")
            print(f"    {info.get('label', name)}")
            print(f"    {info.get('description', '')}")
    print("\n  * = default")


# ── stages ───────────────────────────────────────────────────

def ingest(module: Optional[str] = None) -> None:
    """Bring docs/ into the knowledge base using the chosen ingest module."""
    name = module or defaults()["ingest"]
    print(f"[ingest] using module: {name}")
    load("ingest", name).run()


def index(module: Optional[str] = None, doc: Optional[str] = None) -> None:
    """Build the heading tree for one document, or for the whole corpus."""
    name = module or defaults()["index"]
    print(f"[index] using module: {name}")
    mod = load("index", name)
    if doc:
        mod.build_index(doc)
        return
    docs = sorted(KB_DIR.glob("*.md"))
    if not docs:
        print(f"No documents found in {KB_DIR}/. Run 'pipeline.py ingest' first.")
        sys.exit(1)
    print(f"Building index for {len(docs)} documents...")
    for path in docs:
        mod.build_index(path.stem)
    print("Done.")


def query(text: str, index_module: Optional[str] = None,
          query_module: Optional[str] = None) -> str:
    """Retrieve across the whole index and synthesise an answer."""
    index_name = index_module or defaults()["index"]
    query_name = query_module or defaults()["query"]
    print(f"[query] index={index_name}  synthesis={query_name}")

    index_mod = load("index", index_name)
    index_files = sorted(INDEX_DIR.glob("*.json"))
    if not index_files:
        print(f"No index files in {INDEX_DIR}/. Run 'pipeline.py index' first.")
        sys.exit(1)

    nodes_by_doc: dict = {}
    for idx_file in index_files:
        try:
            nodes = index_mod.retrieve(idx_file.stem, text)
        except Exception as exc:
            print(f"  Warning: retrieval failed for {idx_file.stem}: {exc}")
            continue
        if nodes:
            nodes_by_doc[idx_file.stem] = nodes

    total = sum(len(v) for v in nodes_by_doc.values())
    print(f"\nRetrieved {total} leaf node(s) across {len(nodes_by_doc)} document(s):")
    for doc, nodes in nodes_by_doc.items():
        for node in nodes:
            tag = " [synth]" if getattr(node, "synthetic", False) else ""
            print(f"  {doc} / {node.node_id}{tag}")

    return load("query", query_name).synthesise(text, nodes_by_doc)


# ── CLI ──────────────────────────────────────────────────────

def _resolve_query(text: Optional[str]) -> str:
    text = (text or "").strip() or input("Enter your query: ").strip()
    if not text:
        print("No query provided.")
        sys.exit(1)
    return text


def cmd_list(_args) -> None:
    _print_modules(discover())


def cmd_ingest(args) -> None:
    ingest(args.module)


def cmd_index(args) -> None:
    index(args.module, args.doc)


def cmd_query(args) -> None:
    print("\n" + "=" * 60)
    print(query(_resolve_query(args.query_text), args.index_module, args.query_module))


def cmd_run(args) -> None:
    text = _resolve_query(args.query_text)
    ingest(args.ingest)
    index(args.index)
    print("\n" + "=" * 60)
    print(query(text, args.index, args.query))


# ── main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PageIndex pipeline — select modules per stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    # list
    sub.add_parser("list", help="List available modules for each stage")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Run the ingest stage")
    p_ingest.add_argument("--module", default=None, help="Ingest module name (default: basic_markdown)")

    # index
    p_index = sub.add_parser("index", help="Build the document index")
    p_index.add_argument("--module", default=None, help="Index module name (default: pageindex_custom)")
    p_index.add_argument("--doc",    default=None, help="Single document stem (default: all)")

    # query
    p_query = sub.add_parser("query", help="Retrieve + synthesise answer")
    p_query.add_argument("--index-module", default=None)
    p_query.add_argument("--query-module", default=None)
    p_query.add_argument("query_text", nargs="?", help="Query string (interactive if omitted)")

    # run (full pipeline)
    p_run = sub.add_parser("run", help="Ingest + index + query in one shot")
    p_run.add_argument("--ingest", default=defaults()["ingest"])
    p_run.add_argument("--index",  default=defaults()["index"])
    p_run.add_argument("--query",  default=defaults()["query"])
    p_run.add_argument("query_text", nargs="?")

    args = parser.parse_args()

    dispatch = {
        "list":  cmd_list,
        "ingest": cmd_ingest,
        "index":  cmd_index,
        "query":  cmd_query,
        "run":    cmd_run,
    }

    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
