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

sys.path.insert(0, str(Path(__file__).parent))

from modules.registry import discover, load, defaults


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


# ── sub-commands ─────────────────────────────────────────────

def cmd_list(_args) -> None:
    _print_modules(discover())


def cmd_ingest(args) -> None:
    name = args.module or defaults()["ingest"]
    print(f"[ingest] using module: {name}")
    mod = load("ingest", name)
    mod.run()


def cmd_index(args) -> None:
    name = args.module or defaults()["index"]
    print(f"[index] using module: {name}")
    mod = load("index", name)
    import time

    docs = ([Path(f"knowledge_base/{args.doc}.md")] if args.doc
            else sorted(Path("knowledge_base").glob("*.md")))
    if not docs:
        print("No documents found in knowledge_base/. Run 'pipeline.py ingest' first.")
        sys.exit(1)

    # Indexing is the one expensive step: every node gets an LLM-written
    # summary, once. Report enough that a long run is visibly progressing
    # rather than looking hung.
    print(f"\nBuilding index for {len(docs)} document(s).")
    print("Each node is summarised once by the model; unchanged nodes are reused.")
    print("Legend:  + generated   = reused   ! heuristic fallback\n")

    # Not all index modules take `verbose`; ask the signature rather than
    # catching TypeError, which would swallow a genuine TypeError raised
    # inside build_index and then silently run the whole thing twice.
    import inspect
    try:
        supports_verbose = "verbose" in inspect.signature(mod.build_index).parameters
    except (TypeError, ValueError):
        supports_verbose = False

    started = time.time()
    totals = {"generated": 0, "reused": 0, "heuristic": 0, "errors": 0}
    for i, path in enumerate(docs, 1):
        print(f"  [{i}/{len(docs)}] {path.stem}")
        report = (mod.build_index(path.stem, verbose=True) if supports_verbose
                  else mod.build_index(path.stem)) or {}
        totals["generated"] += report.get("summaries_generated", 0)
        totals["reused"]    += report.get("summaries_reused", 0)
        totals["heuristic"] += report.get("summaries_heuristic", 0)
        totals["errors"]    += len(report.get("errors") or [])
        print(f"      done in {report.get('seconds', 0)}s — "
              f"{report.get('summaries_generated', 0)} generated, "
              f"{report.get('summaries_reused', 0)} reused\n", flush=True)

    elapsed = time.time() - started
    print(f"Done in {elapsed:.0f}s — {totals['generated']} summaries generated, "
          f"{totals['reused']} reused, {totals['heuristic']} heuristic fallback(s).")
    if totals["errors"]:
        print(f"  {totals['errors']} summariser error(s); those nodes fell back to "
              f"the heuristic and will be retried on the next index run.")
    print("Re-running this on unchanged documents costs nothing.")


def cmd_query(args) -> None:
    query = args.query_text
    if not query:
        query = input("Enter your query: ").strip()
    if not query:
        print("No query provided.")
        sys.exit(1)

    index_name = args.index_module or defaults()["index"]
    query_name = args.query_module or defaults()["query"]

    print(f"[query] index={index_name}  synthesis={query_name}")
    index_mod = load("index", index_name)

    from pathlib import Path as _Path
    index_dir = getattr(index_mod, "INDEX_DIR", _Path("index"))
    index_files = sorted(index_dir.glob("*.json"))
    if not index_files:
        print(f"No index files in {index_dir}/. Run 'pipeline.py index' first.")
        sys.exit(1)

    import json
    nodes_by_doc: dict = {}
    for idx_file in index_files:
        doc_name = idx_file.stem
        try:
            nodes = index_mod.retrieve(doc_name, query)
        except Exception as exc:
            print(f"  Warning: retrieval failed for {doc_name}: {exc}")
            continue
        if nodes:
            nodes_by_doc[doc_name] = nodes

    total = sum(len(v) for v in nodes_by_doc.values())
    print(f"\nRetrieved {total} leaf node(s) across {len(nodes_by_doc)} document(s):")
    for doc, nodes in nodes_by_doc.items():
        for n in nodes:
            tag = " [synth]" if getattr(n, "synthetic", False) else ""
            print(f"  {doc} / {n.node_id}{tag}")

    print("\n" + "=" * 60)
    query_mod = load("query", query_name)
    answer = query_mod._synthesise(query, nodes_by_doc)
    print(answer)


def cmd_run(args) -> None:
    cmd_ingest(type("A", (), {"module": args.ingest})())
    cmd_index(type("A", (), {"module": args.index, "doc": None})())
    args2 = type("A", (), {
        "query_text": args.query_text,
        "index_module": args.index,
        "query_module": args.query,
    })()
    cmd_query(args2)


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
