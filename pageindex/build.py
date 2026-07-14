"""Building a document's heading tree.

Deterministic: no model is consulted. Headings become the hierarchy, text before
the first child becomes a synthetic leaf, summaries are heuristic, and pins are
lifted out of the markdown onto the node.
"""

import json

from paths import INDEX_DIR, KB_DIR

from .nodes import (
    _attach_pins,
    _collect_leaves,
    _build_tree,
    _node_to_dict,
    _parse_headings,
    _populate_content,
    _populate_summaries,
    _promote_preambles,
)


def build_index(doc_name: str) -> None:
    """Parse and index a single document from knowledge_base/."""
    doc_path = KB_DIR / f"{doc_name}.md"
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    print(f"  Indexing {doc_name}...")
    text = doc_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    headings = _parse_headings(text)
    nodes = _build_tree(headings)
    _attach_pins(nodes, lines)
    _promote_preambles(nodes, lines)
    _populate_content(nodes, lines)
    _populate_summaries(nodes)

    INDEX_DIR.mkdir(exist_ok=True)
    index_path = INDEX_DIR / f"{doc_name}.json"
    index_path.write_text(
        json.dumps([_node_to_dict(n) for n in nodes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    leaves = _collect_leaves(nodes)
    print(f"    → {index_path} ({len(nodes)} top-level nodes, {len(leaves)} leaves)")
