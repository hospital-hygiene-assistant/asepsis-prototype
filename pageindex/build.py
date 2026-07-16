"""Building a document's heading tree.

Deterministic: no model is consulted. Headings become the hierarchy, text before
the first child becomes a synthetic leaf, summaries are heuristic, and pins are
lifted out of the markdown onto the node.
"""

import json

from paths import INDEX_DIR, KB_DIR

from .generations import IndexGenerationStore, IndexSnapshot
from .nodes import (
    _collect_leaves,
    _node_to_dict,
    parse_document,
)


def _build_document(doc_name: str) -> tuple[str, int, int]:
    doc_path = KB_DIR / f"{doc_name}.md"
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")
    nodes = parse_document(doc_path.read_text(encoding="utf-8"), doc_name)
    payload = json.dumps(
        [_node_to_dict(node) for node in nodes], indent=2, ensure_ascii=False
    )
    return payload, len(nodes), len(_collect_leaves(nodes))


def build_index(doc_name: str) -> None:
    """Parse and index a single document from knowledge_base/."""
    print(f"  Indexing {doc_name}...")
    payload, top_level_count, leaf_total = _build_document(doc_name)

    INDEX_DIR.mkdir(exist_ok=True)
    index_path = INDEX_DIR / f"{doc_name}.json"
    index_path.write_text(payload, encoding="utf-8")

    print(
        f"    → {index_path} ({top_level_count} top-level nodes, "
        f"{leaf_total} leaves)"
    )


def build_generation(doc_names: list[str]) -> IndexSnapshot:
    """Build and atomically promote one complete immutable corpus index."""
    documents: dict[str, str] = {}
    for doc_name in sorted(doc_names):
        print(f"  Indexing {doc_name}...")
        payload, top_level_count, leaf_total = _build_document(doc_name)
        documents[doc_name] = payload
        print(
            f"    staged {doc_name} ({top_level_count} top-level nodes, "
            f"{leaf_total} leaves)"
        )
    snapshot = IndexGenerationStore(INDEX_DIR).publish(documents)
    print(f"    → promoted index generation {snapshot.generation_id}")
    return snapshot
