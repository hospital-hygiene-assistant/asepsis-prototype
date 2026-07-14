"""Running a retrieval pass over the whole index.

Shared by /api/run and /api/chat so both drive the same live progress state and
return the same shape.
"""

import pageindex as _pi
from paths import INDEX_DIR

from .trees import leaf_count, read_tree


def run_retrieval(query: str, index_mod, run) -> dict:
    """Two-phase retrieval across every indexed document.

    Returns {doc: {tree, retrieved_ids, node_meta, nodes}}. Documents whose
    index has gone missing are skipped rather than failing the run.
    """
    index_files = sorted(INDEX_DIR.glob("*.json"))
    # Progress polling needs a denominator before the first leaf is evaluated.
    run.state.start(sum(leaf_count(read_tree(idx)) for idx in index_files))

    results = {}
    for idx in index_files:
        doc_name = idx.stem
        try:
            retrieve_fn = getattr(index_mod, "retrieve_with_metadata", None)
            if retrieve_fn:
                nodes, node_meta = retrieve_fn(doc_name, query, run.state)
            else:
                nodes, node_meta = index_mod.retrieve(doc_name, query, run.state), {}
        except FileNotFoundError:
            continue

        results[doc_name] = {
            "tree": read_tree(idx),
            "retrieved_ids": [n.node_id for n in nodes],
            "node_meta": node_meta,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "title": n.title,
                    "content": n.content or "",
                    "synthetic": n.synthetic,
                    "heading_level": n.heading_level,
                    "summary": n.summary,
                    "pin": n.pin,
                    "reason": (node_meta.get(n.node_id) or {}).get("reason", ""),
                    "quote": (node_meta.get(n.node_id) or {}).get("quote", ""),
                }
                for n in nodes
            ],
        }
    return results


def count_eval_errors(results: dict) -> int:
    """Leaves the model could not evaluate, across every document in a run.

    Distinct from rejected leaves: these were never actually checked.
    """
    return sum(
        1
        for doc in results.values()
        for meta in (doc.get("node_meta") or {}).values()
        if meta.get("status") == "error"
    )


def breadcrumbs_for(results: dict) -> dict:
    """{doc: {node_id: 'Doc › Section › Leaf'}} for every retrieved node."""
    crumbs: dict[str, dict] = {}
    for doc_name, doc in results.items():
        nodes = [_pi._node_from_dict(d) for d in doc["tree"]]
        nodes_by_id = _pi._build_nodes_by_id(nodes)
        parent_map = _pi._build_parent_map(nodes)
        crumbs[doc_name] = {
            nid: _pi._make_breadcrumb(nid, parent_map, nodes_by_id)
            for nid in doc["retrieved_ids"]
        }
    return crumbs
