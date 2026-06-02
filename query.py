"""
Module 3: Query-Answer
Retrieves all relevant PageIndex leaf nodes across all indexed documents,
displays them with full content, then synthesises a final answer via Ollama.

Usage:
    python query.py "What are first-line antihypertensives?"
    python query.py   # interactive prompt
"""

import argparse
import sys
from pathlib import Path

import ollama

from pageindex import retrieve, PageNode, INDEX_DIR, MODEL as DEFAULT_MODEL

SYNTHESIS_PROMPT_TEMPLATE = """\
You are a medical knowledge assistant. Answer the following query using ONLY the
provided source passages. Cite each claim with the source reference in brackets.

Query: QUERY_PLACEHOLDER

Source passages:
PASSAGES_PLACEHOLDER

Instructions:
- Be concise and clinically accurate.
- Cite sources inline using the reference tags provided.
- After your answer, list the sources you used.
- If the passages do not contain enough information to answer fully, say so.
"""

MODEL = DEFAULT_MODEL


def _format_node(node: PageNode, doc_name: str) -> str:
    synthetic_tag = " [synthetic]" if node.synthetic else ""
    ref = f"[{doc_name} / {node.node_id}{synthetic_tag}]"
    return f"{ref}\n{node.title}\n{'─' * 60}\n{node.content or '(no content)'}"


def _synthesise(query: str, nodes_by_doc: dict[str, list[PageNode]]) -> str:
    passages = []
    for doc_name, nodes in nodes_by_doc.items():
        for node in nodes:
            ref = f"[{doc_name} / {node.node_id}]"
            passages.append(f"{ref}\n{node.content or ''}")

    if not passages:
        return "No relevant passages found."

    prompt = SYNTHESIS_PROMPT_TEMPLATE.replace("QUERY_PLACEHOLDER", query).replace(
        "PASSAGES_PLACEHOLDER", "\n\n".join(passages)
    )
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"]


def run_query(query: str, model: str = DEFAULT_MODEL) -> None:
    global MODEL
    MODEL = model

    index_files = sorted(INDEX_DIR.glob("*.json"))
    if not index_files:
        print(f"No index files in {INDEX_DIR}/. Run pageindex.py first.")
        sys.exit(1)

    print(f"\nQuery: {query}")
    print("=" * 70)

    nodes_by_doc: dict[str, list[PageNode]] = {}
    for idx_file in index_files:
        doc_name = idx_file.stem
        try:
            nodes = retrieve(doc_name, query)
        except Exception as e:
            print(f"  Warning: retrieval failed for {doc_name}: {e}")
            continue
        if nodes:
            nodes_by_doc[doc_name] = nodes

    total = sum(len(v) for v in nodes_by_doc.values())
    if total == 0:
        print("No relevant nodes found across any document.")
        return

    print(f"\nRetrieved {total} leaf node(s) across {len(nodes_by_doc)} document(s):\n")
    for doc_name, nodes in nodes_by_doc.items():
        print(f"  {doc_name}")
        for n in nodes:
            tag = " [synth]" if n.synthetic else ""
            print(f"    • {n.node_id}{tag}: {n.title}")

    print("\n" + "=" * 70)
    print("RETRIEVED PASSAGES")
    print("=" * 70)
    for doc_name, nodes in nodes_by_doc.items():
        for node in nodes:
            print()
            print(_format_node(node, doc_name))

    print("\n" + "=" * 70)
    print("SYNTHESISED ANSWER")
    print("=" * 70 + "\n")
    print(_synthesise(query, nodes_by_doc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the PageIndex knowledge base.")
    parser.add_argument("query", nargs="?", help="Query string (interactive if omitted).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    query = args.query or input("Enter your query: ").strip()
    if not query:
        print("No query provided.")
        sys.exit(1)

    run_query(query, model=args.model)


if __name__ == "__main__":
    main()
