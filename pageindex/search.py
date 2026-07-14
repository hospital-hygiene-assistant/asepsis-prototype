"""Retrieval: prune the tree with the model, then judge the surviving leaves.

Two phases. A section whose outline looks irrelevant is pruned with its
descendants; every leaf that survives is evaluated on its own content and has to
come back with a verbatim quote to count as relevant.

Both phases fail open. A section the model could not judge is kept, and a leaf it
could not judge is recorded as errored rather than rejected, because "not
checked" is not "irrelevant".
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import ollama

from paths import INDEX_DIR

from . import run_state
from .clients import acquire, pool, release
from .llm import _chat, _parse_json_response
from .nodes import (
    PageNode,
    _collect_leaves,
    _build_nodes_by_id,
    _build_parent_map,
    _make_breadcrumb,
    _node_from_dict,
)
from .prompts import EXPLAIN_PROMPT, LEAF_EVAL_PROMPT, SECTION_CHECK_PROMPT

# ---------------------------------------------------------------------------
# Breadcrumb helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-leaf evaluation
# ---------------------------------------------------------------------------

def _evaluate_leaf(
    leaf: "PageNode",
    query: str,
    doc_name: str,
    breadcrumb: str,
    parent_summary: str,
    client: Optional[ollama.Client] = None,
    client_url: str = "",
) -> tuple[str, dict]:
    """Evaluate one leaf. Returns (node_id, {relevant, reason, quote, status})."""
    prompt = (
        LEAF_EVAL_PROMPT
        .replace("QUERY_PLACEHOLDER", query)
        .replace("DOCNAME_PLACEHOLDER", doc_name.replace("_", " ").title())
        .replace("BREADCRUMB_PLACEHOLDER", breadcrumb)
        .replace("PARENT_SUMMARY_PLACEHOLDER", parent_summary or "top-level section")
        .replace("CONTENT_PLACEHOLDER", leaf.content or "(no content)")
    )
    acquire(client_url)
    try:
        raw = _chat(prompt, client, client_url)
        result = _parse_json_response(raw)
        
        if isinstance(result, dict):
            relevant = bool(result.get("relevant"))
            reason = str(result.get("reason") or "").strip()
            quote = str(result.get("quote") or "").strip()
            
            if relevant:
                run_state.mark_retrieved(leaf.node_id)
                run_state.set_meta(leaf.node_id, "retrieved", reason, quote)
                return leaf.node_id, {
                    "relevant": True,
                    "reason": reason,
                    "quote": quote,
                    "status": "retrieved"
                }
            else:
                run_state.mark_rejected(leaf.node_id)
                run_state.set_meta(leaf.node_id, "rejected", reason)
                return leaf.node_id, {
                    "relevant": False,
                    "reason": reason,
                    "quote": "",
                    "status": "rejected"
                }
        failure = "the model returned no usable verdict"
    except Exception as exc:
        print(f"    [warn] leaf eval failed for {leaf.node_id}: {exc}", file=sys.stderr)
        failure = str(exc)
    finally:
        release(client_url)

    # Reached only when the model could not be consulted, or answered with
    # something unparseable. That is not a judgement that the passage is
    # irrelevant, and must not be recorded as one: a caller seeing every leaf
    # "rejected" would report "nothing relevant was found" to a clinician when
    # the truth is that nothing was actually checked. Mirrors the section
    # check, which already refuses to prune on error.
    run_state.mark_errored(leaf.node_id)
    run_state.set_meta(leaf.node_id, "error", failure)
    return leaf.node_id, {
        "relevant": False,
        "reason": failure,
        "quote": "",
        "status": "error"
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _format_descendant_outline(node: "PageNode", depth: int = 0) -> str:
    """Indented bullet list of all descendant headings — gives the LLM full visibility."""
    lines: list[str] = []
    for child in node.children:
        prefix = "  " * depth
        marker = "•" if not child.is_leaf else "-"
        lines.append(f"{prefix}{marker} {child.title}")
        if child.children:
            lines.append(_format_descendant_outline(child, depth + 1))
    return "\n".join(l for l in lines if l)


def _check_section_relevant(
    node: "PageNode",
    query: str,
    breadcrumb: str,
    client: Optional[ollama.Client] = None,
    client_url: str = "",
) -> tuple[bool, str]:
    """Lightweight LLM call: can this section contain a direct answer?"""
    descendants = _format_descendant_outline(node) or "(no subsections)"
    prompt = (
        SECTION_CHECK_PROMPT
        .replace("QUERY_PLACEHOLDER", query)
        .replace("BREADCRUMB_PLACEHOLDER", breadcrumb)
        .replace("DESCENDANTS_PLACEHOLDER", descendants)
    )
    acquire(client_url)
    try:
        raw = _chat(prompt, client, client_url)
        result = _parse_json_response(raw)
        verdict = isinstance(result, dict) and bool(result.get("relevant"))
        reason = ""
        if isinstance(result, dict):
            reason = str(result.get("reason") or "").strip()
        
        if verdict:
            run_state.mark_kept(node.node_id)
            run_state.set_meta(node.node_id, "kept", reason)

        print(f"    [prune-check] {node.node_id}: {'KEEP' if verdict else 'PRUNE'} (raw={raw!r:.120})", file=sys.stderr)
        return verdict, reason
    except Exception as exc:
        run_state.mark_kept(node.node_id)
        print(f"    [prune-check] {node.node_id}: ERROR ({exc}) — keeping", file=sys.stderr)
        return True, f"Error checking section: {exc}"  # conservative: never prune on error
    finally:
        release(client_url)


def make_client(url: str) -> ollama.Client:
    """Build an Ollama client for a specific instance URL (used by the explainer)."""
    return ollama.Client(host=url)


def explain_nonselection(
    node: "PageNode",
    query: str,
    doc_name: str,
    breadcrumb: str,
    client: Optional[ollama.Client] = None,
    client_url: str = "",
) -> dict:
    """On-demand: read the actual content and explain (grounded) what this
    non-selected section covers and why it doesn't answer the query.

    For a section with no own content, concatenate descendant leaf text so the
    judgement is anchored to real content rather than the heading alone.
    """
    content = node.content
    if not content:
        leaves = _collect_leaves([node])
        content = "\n\n".join((l.title + "\n" + (l.content or "")) for l in leaves).strip()
    content = (content or "(no content)")[:4000]

    prompt = (
        EXPLAIN_PROMPT
        .replace("QUERY_PLACEHOLDER", query)
        .replace("DOCNAME_PLACEHOLDER", doc_name.replace("_", " ").title())
        .replace("BREADCRUMB_PLACEHOLDER", breadcrumb)
        .replace("CONTENT_PLACEHOLDER", content)
    )
    try:
        raw = _chat(prompt, client, client_url)
        result = _parse_json_response(raw)
        if isinstance(result, dict):
            return {
                "topic": str(result.get("topic") or "").strip(),
                "addresses_query": bool(result.get("addresses_query")),
                "reason": str(result.get("reason") or "").strip(),
            }
    except Exception as exc:
        print(f"    [explain] failed for {node.node_id}: {exc}", file=sys.stderr)
    return {"topic": "", "addresses_query": False, "reason": ""}


def _collect_all_nodes(nodes: list["PageNode"]) -> list["PageNode"]:
    result: list["PageNode"] = []
    for node in nodes:
        result.append(node)
        result.extend(_collect_all_nodes(node.children))
    return result


def _prune_and_collect(
    nodes: list["PageNode"],
    query: str,
    parent_map: dict,
    nodes_by_id: dict,
    node_assignment: dict[str, tuple],
    node_meta: dict[str, dict],
) -> list["PageNode"]:
    """
    BFS top-down pruning. At each level, check all internal nodes in parallel;
    recurse only into relevant ones. Leaves always survive to full evaluation.
    Returns the list of leaf candidates that passed pruning.
    """
    candidate_leaves: list["PageNode"] = []
    frontier = list(nodes)

    while frontier:
        sections = [n for n in frontier if not n.is_leaf]
        candidate_leaves.extend(n for n in frontier if n.is_leaf)

        if not sections:
            break

        with ThreadPoolExecutor(max_workers=len(sections)) as workers:
            futures = {
                workers.submit(
                    _check_section_relevant,
                    node, query,
                    _make_breadcrumb(node.node_id, parent_map, nodes_by_id),
                    *node_assignment.get(node.node_id, pool()[0]),
                ): node
                for node in sections
            }
            next_frontier: list["PageNode"] = []
            for future in as_completed(futures):
                node = futures[future]
                verdict, reason = future.result()
                
                node_meta[node.node_id] = {
                    "relevant": verdict,
                    "reason": reason,
                    "status": "kept" if verdict else "pruned"
                }

                if verdict:
                    next_frontier.extend(node.children)
                else:
                    # Pruned: count all leaf descendants toward "done" so the
                    # progress counter reaches total. They won't be evaluated again.
                    _prune_reason = f"Pruned because ancestor section '{node.title}' was pruned."
                    run_state.set_meta(node.node_id, "pruned", reason or _prune_reason)
                    for leaf in _collect_leaves([node]):
                        run_state.leaf_done()
                        run_state.mark_pruned(leaf.node_id)
                        run_state.set_meta(leaf.node_id, "pruned", _prune_reason)
                        node_meta[leaf.node_id] = {
                            "relevant": False,
                            "reason": _prune_reason,
                            "status": "pruned"
                        }
                    for descendant in _collect_all_nodes([node]):
                        run_state.mark_pruned(descendant.node_id)
                        if descendant.node_id != node.node_id:
                            run_state.set_meta(descendant.node_id, "pruned", _prune_reason)
                            node_meta[descendant.node_id] = {
                                "relevant": False,
                                "reason": _prune_reason,
                                "status": "pruned"
                            }

        frontier = next_frontier

    return candidate_leaves


def retrieve(doc_name: str, query: str) -> list[PageNode]:
    """Return relevant leaf nodes for a query against one document's index."""
    nodes_result, _ = retrieve_with_metadata(doc_name, query)
    return nodes_result


def retrieve_with_metadata(doc_name: str, query: str) -> tuple[list[PageNode], dict[str, dict]]:
    """
    Two-phase retrieval with top-down pruning.
    Phase 1: BFS section checks — prune branches whose sections are irrelevant.
    Phase 2: Full leaf evaluation (reason + quote) on survivors only.
    Returns (selected_leaves_in_doc_order, {node_id: {reason, quote}}).
    """
    index_path = INDEX_DIR / f"{doc_name}.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}. Run build first.")

    nodes = [_node_from_dict(d) for d in json.loads(index_path.read_text(encoding="utf-8"))]
    leaves = _collect_leaves(nodes)

    if not leaves:
        return [], {}

    parent_map  = _build_parent_map(nodes)
    nodes_by_id = _build_nodes_by_id(nodes)

    # Assign every node (section + leaf) to an Ollama instance round-robin by branch.
    branch_roots = nodes[0].children if (len(nodes) == 1 and nodes[0].children) else nodes
    node_assignment: dict[str, tuple[ollama.Client, str]] = {}
    for i, branch in enumerate(branch_roots):
        client, url = pool()[i % len(pool())]
        for node in _collect_all_nodes([branch]):
            node_assignment[node.node_id] = (client, url)
    # Fallback for any node not covered (e.g., single-root flat doc)
    for node in _collect_all_nodes(nodes):
        if node.node_id not in node_assignment:
            node_assignment[node.node_id] = pool()[0]

    node_meta: dict[str, dict] = {}
    if nodes:
        node_meta[nodes[0].node_id] = {
            "relevant": True,
            "reason": f"Document root '{nodes[0].title}' — starting point for retrieval.",
            "status": "kept"
        }

    # Phase 1 — top-down pruning
    top = nodes[0].children if (len(nodes) == 1 and nodes[0].children) else nodes
    surviving = _prune_and_collect(top, query, parent_map, nodes_by_id, node_assignment, node_meta)
    print(
        f"  [prune] {doc_name}: {len(surviving)}/{len(leaves)} leaves after pruning",
        file=sys.stderr,
    )

    # Phase 2 — full leaf evaluation on survivors
    if surviving:
        with ThreadPoolExecutor(max_workers=len(surviving)) as workers:
            futures = {
                workers.submit(
                    _evaluate_leaf,
                    leaf,
                    query,
                    doc_name,
                    _make_breadcrumb(leaf.node_id, parent_map, nodes_by_id),
                    parent_map[leaf.node_id].summary if leaf.node_id in parent_map else "",
                    *node_assignment.get(leaf.node_id, pool()[0]),
                ): leaf
                for leaf in surviving
            }
            for future in as_completed(futures):
                node_id, meta = future.result()
                node_meta[node_id] = meta
                run_state.leaf_done()

    # Preserve original document order
    selected = [l for l in leaves if node_meta.get(l.node_id, {}).get("relevant")]
    return selected, node_meta
