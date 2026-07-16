"""Local corpus pipeline.

Commands:
  python3 pipeline.py list
  python3 pipeline.py ingest [--module basic_markdown]
  python3 pipeline.py index [--doc NAME]
  python3 pipeline.py query "your question"
  python3 pipeline.py run [--ingest basic_markdown] "your question"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tauri-app"))

import pageindex
from api.answering_runtime import answer_question
from api.question_answering import AnswerKind, Question
from pageindex import QuestionRun
from modules.registry import defaults, discover, load
from paths import KB_DIR, LIBRARY_DIR


def _print_modules(registry: dict) -> None:
    print("\nIngest adapters:")
    for name, info in registry.get("ingest", {}).items():
        marker = " *" if name == defaults()["ingest"] else "  "
        print(f"{marker} {name}")
        print(f"    {info.get('label', name)}")
        print(f"    {info.get('description', '')}")
    print("\n  * = default")


def ingest(module: Optional[str] = None) -> None:
    """Bring source documents into the knowledge base through one adapter."""
    name = module or defaults()["ingest"]
    print(f"[ingest] using adapter: {name}")
    load("ingest", name).run()


def index(doc: Optional[str] = None) -> None:
    """Atomically publish one Expected library generation for the corpus."""
    documents = [path.stem for path in sorted(KB_DIR.glob("*.md"))]
    if not documents:
        print(f"No documents found in {KB_DIR}/. Run 'pipeline.py ingest' first.")
        raise SystemExit(1)
    if doc and doc not in documents:
        print(f"Document '{doc}' is not present in {KB_DIR}/.")
        raise SystemExit(1)
    print(f"Building one generation for {len(documents)} documents...")
    pageindex.build_generation(documents)
    print("Done.")


def query(text: str) -> str:
    """Answer through the same whole-library interface as practitioner chat."""
    run = QuestionRun("cli")
    outcome = answer_question(Question(text), run)
    if (
        outcome.kind is AnswerKind.RETRIEVAL_UNAVAILABLE
        and any(
            diagnostic.code == "expected_library_unavailable"
            for diagnostic in outcome.search.diagnostics
        )
    ):
        print(
            f"Expected library unavailable in {LIBRARY_DIR}/. "
            "Run 'pipeline.py index' first."
        )
        raise SystemExit(1)
    print(
        f"\nVerified {len(outcome.search.evidence)} source passage(s) across "
        f"{len(outcome.search.documents)} document(s):"
    )
    for evidence in outcome.search.evidence:
        print(f"  {evidence.document} / {evidence.node.node_id}")

    if outcome.kind is AnswerKind.ANSWERED and outcome.answer is not None:
        return outcome.answer.content
    if outcome.kind is AnswerKind.INSUFFICIENT_EVIDENCE:
        return AnswerKind.INSUFFICIENT_EVIDENCE.value
    raise RuntimeError(f"question answering did not complete: {outcome.kind.value}")


def _resolve_query(text: Optional[str]) -> str:
    text = (text or "").strip() or input("Enter your query: ").strip()
    if not text:
        print("No query provided.")
        raise SystemExit(1)
    return text


def cmd_list(_args) -> None:
    _print_modules(discover())


def cmd_ingest(args) -> None:
    ingest(args.module)


def cmd_index(args) -> None:
    index(doc=args.doc)


def cmd_query(args) -> None:
    print("\n" + "=" * 60)
    print(query(_resolve_query(args.query_text)))


def cmd_run(args) -> None:
    text = _resolve_query(args.query_text)
    ingest(args.ingest)
    index()
    print("\n" + "=" * 60)
    print(query(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="ASEPSIS local corpus pipeline")
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.add_parser("list", help="List ingest adapters")

    p_ingest = sub.add_parser("ingest", help="Run ingest")
    p_ingest.add_argument("--module", default=None, help="Ingest adapter name")

    p_index = sub.add_parser("index", help="Build one immutable index generation")
    p_index.add_argument("--doc", default=None, help="Validate that a document is present")

    p_query = sub.add_parser("query", help="Retrieve and answer")
    p_query.add_argument("query_text", nargs="?", help="Question (interactive if omitted)")

    p_run = sub.add_parser("run", help="Ingest, index, and answer")
    p_run.add_argument("--ingest", default=defaults()["ingest"])
    p_run.add_argument("query_text", nargs="?")

    args = parser.parse_args()
    dispatch = {
        "list": cmd_list,
        "ingest": cmd_ingest,
        "index": cmd_index,
        "query": cmd_query,
        "run": cmd_run,
    }
    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
