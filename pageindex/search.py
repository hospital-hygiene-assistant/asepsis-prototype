"""Retrieval: prune the tree with the model, then judge the surviving leaves.

Two phases. A section whose outline looks irrelevant is pruned with its
descendants; every leaf that survives is evaluated on its own content and has to
come back with a verbatim quote to count as relevant.

Both phases fail open. A section the model could not judge is kept, and a leaf it
could not judge is recorded as errored rather than rejected, because "not
checked" is not "irrelevant".
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import ollama

from .question_run import QuestionRun
from .document_index import DocumentIndex
from .clients import acquire, pool, release
from .llm import _chat, _parse_json_response
from .nodes import PageNode
from .prompts import EXPLAIN_PROMPT, LEAF_EVAL_PROMPT, SECTION_CHECK_PROMPT
from .settings import settings

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
    model: str | None = None,
    *,
    run: QuestionRun,
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
    failure_code = "evaluation_error"
    acquire(client_url)
    try:
        raw = _chat(prompt, client, client_url, model)
        result = _parse_json_response(raw)

        if isinstance(result, dict):
            relevant = bool(result.get("relevant"))
            reason = str(result.get("reason") or "").strip()
            quote = str(result.get("quote") or "").strip()

            if relevant:
                if not quote or quote not in (leaf.content or ""):
                    failure = (
                        "the model returned a relevant verdict without a "
                        "non-empty exact quote"
                    )
                    failure_code = "invalid_quote"
                else:
                    run.mark("retrieved", leaf.node_id)
                    run.record(leaf.node_id, "retrieved", reason, quote)
                    return leaf.node_id, {
                        "relevant": True,
                        "reason": reason,
                        "quote": quote,
                        "status": "retrieved"
                    }
            else:
                run.mark("rejected", leaf.node_id)
                run.record(leaf.node_id, "rejected", reason)
                return leaf.node_id, {
                    "relevant": False,
                    "reason": reason,
                    "quote": "",
                    "status": "rejected"
                }
        else:
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
    run.mark("errored", leaf.node_id)
    run.record(leaf.node_id, "error", failure)
    return leaf.node_id, {
        "relevant": False,
        "reason": failure,
        "quote": "",
        "status": "error",
        "code": failure_code,
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


def _terms(text: str) -> set[str]:
    return {
        term for term in re.findall(r"\w+", text.casefold()) if len(term) >= 4
    }


def _heading_terms(node: "PageNode", index: DocumentIndex) -> set[str]:
    titles = [node.title]
    titles.extend(item.title for item in index.descendants(node.node_id))
    return set().union(*(_terms(title) for title in titles))


def _check_section_relevant(
    node: "PageNode",
    query: str,
    breadcrumb: str,
    index: DocumentIndex,
    client: Optional[ollama.Client] = None,
    client_url: str = "",
    model: str | None = None,
    *,
    run: QuestionRun,
) -> tuple[bool, str]:
    """Lightweight LLM call: can this section contain a direct answer?"""
    query_terms = _terms(query)
    direct_matches = sorted(query_terms & _heading_terms(node, index))
    if direct_matches:
        reason = (
            "Kept because the query directly matches descendant heading term(s): "
            + ", ".join(direct_matches)
        )
        run.mark("kept", node.node_id)
        run.record(node.node_id, "kept", reason)
        print(
            f"    [prune-check] {node.node_id}: KEEP (heading match: "
            f"{', '.join(direct_matches)})",
            file=sys.stderr,
        )
        return True, reason

    descendants = _format_descendant_outline(node) or "(no subsections)"
    prompt = (
        SECTION_CHECK_PROMPT
        .replace("QUERY_PLACEHOLDER", query)
        .replace("BREADCRUMB_PLACEHOLDER", breadcrumb)
        .replace("DESCENDANTS_PLACEHOLDER", descendants)
    )
    acquire(client_url)
    try:
        raw = _chat(prompt, client, client_url, model)
        result = _parse_json_response(raw)
        if not isinstance(result, dict):
            raise ValueError("the model returned no usable section verdict")
        verdict = bool(result.get("relevant"))
        reason = str(result.get("reason") or "").strip()

        if verdict:
            run.mark("kept", node.node_id)
            run.record(node.node_id, "kept", reason)

        print(f"    [prune-check] {node.node_id}: {'KEEP' if verdict else 'PRUNE'} (raw={raw!r:.120})", file=sys.stderr)
        return verdict, reason
    except Exception as exc:
        run.mark("kept", node.node_id)
        run.record(node.node_id, "error", str(exc))
        print(f"    [prune-check] {node.node_id}: ERROR ({exc}) — keeping", file=sys.stderr)
        return True, f"Error checking section: {exc}"  # conservative: never prune on error
    finally:
        release(client_url)


def explain_nonselection(
    index: DocumentIndex,
    node_id: str,
    query: str,
    doc_name: str,
    client: Optional[ollama.Client] = None,
    client_url: str = "",
) -> dict:
    """On-demand: read the actual content and explain (grounded) what this
    non-selected section covers and why it doesn't answer the query.

    For a section with no own content, concatenate descendant leaf text so the
    judgement is anchored to real content rather than the heading alone.
    """
    node = index.node(node_id)
    breadcrumb = index.breadcrumb(node_id)
    content = node.content
    if not content:
        leaves = index.leaves_under(node_id)
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


def _prune_and_collect(
    nodes: list["PageNode"],
    query: str,
    index: DocumentIndex,
    node_assignment: dict[str, tuple],
    node_meta: dict[str, dict],
    run: QuestionRun,
    model: str | None = None,
) -> list["PageNode"]:
    """
    BFS top-down pruning. At each level, check all internal nodes in parallel;
    recurse only into relevant ones. Leaves always survive to full evaluation.
    Returns the list of leaf candidates that passed pruning.
    """
    default_instance = pool()[0]
    model = model or settings.model
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
                    index.breadcrumb(node.node_id),
                    index,
                    *node_assignment.get(node.node_id, default_instance),
                    model,
                    run=run,
                ): node
                for node in sections
            }
            next_frontier: list["PageNode"] = []
            for future in as_completed(futures):
                node = futures[future]
                verdict, reason = future.result()
                recorded = run.events()["meta"].get(node.node_id, {})
                status = "error" if recorded.get("status") == "error" else (
                    "kept" if verdict else "pruned"
                )
                node_meta[node.node_id] = {
                    "relevant": verdict,
                    "reason": reason,
                    "status": status,
                }

                if verdict:
                    next_frontier.extend(node.children)
                else:
                    # Pruned: count all leaf descendants toward "done" so the
                    # progress counter reaches total. They won't be evaluated again.
                    _prune_reason = f"Pruned because ancestor section '{node.title}' was pruned."
                    run.record(node.node_id, "pruned", reason or _prune_reason)
                    for leaf in index.leaves_under(node.node_id):
                        run.leaf_complete()
                        run.mark("pruned", leaf.node_id)
                        run.record(leaf.node_id, "pruned", _prune_reason)
                        node_meta[leaf.node_id] = {
                            "relevant": False,
                            "reason": _prune_reason,
                            "status": "pruned"
                        }
                    for descendant in index.descendants(
                        node.node_id, include_self=True
                    ):
                        run.mark("pruned", descendant.node_id)
                        if descendant.node_id != node.node_id:
                            run.record(descendant.node_id, "pruned", _prune_reason)
                            node_meta[descendant.node_id] = {
                                "relevant": False,
                                "reason": _prune_reason,
                                "status": "pruned"
                            }

        frontier = next_frontier

    return candidate_leaves


def retrieve_with_metadata(
    doc_name: str,
    query: str,
    run: Optional[QuestionRun],
    index: DocumentIndex,
    model: str | None = None,
    instances=None,
) -> tuple[list[PageNode], dict[str, dict]]:
    """Two-phase retrieval with top-down pruning.

    Phase 1: BFS section checks — prune branches whose sections are irrelevant.
    Phase 2: full leaf evaluation (reason + quote) on the survivors only.

    Records into `run` as it goes, so a client can watch. Without one, the
    verdicts are still returned; only the live view is skipped.

    Returns (selected_leaves_in_doc_order, {node_id: {reason, quote}}).
    """
    run = run or QuestionRun()
    retrieval_model = model or settings.model
    nodes = list(index.nodes)
    leaves = list(index.leaves)

    if not leaves:
        return [], {}
    if run.progress()["total"] == 0:
        run.set_total(len(leaves))

    # Assign every node (section + leaf) to an Ollama instance round-robin by
    # branch. Read the pool once: it can be swapped while a run is in flight,
    # and a branch has to keep the instance its nodes were assigned to.
    instances = tuple(instances) if instances is not None else tuple(pool())
    branch_roots = nodes[0].children if (len(nodes) == 1 and nodes[0].children) else nodes
    node_assignment: dict[str, tuple[ollama.Client, str]] = {}
    for i, branch in enumerate(branch_roots):
        client, url = instances[i % len(instances)]
        for node in index.descendants(branch.node_id, include_self=True):
            node_assignment[node.node_id] = (client, url)
    # Fallback for any node not covered (e.g., single-root flat doc)
    for node in index.all_nodes:
        node_assignment.setdefault(node.node_id, instances[0])

    node_meta: dict[str, dict] = {}
    if nodes:
        node_meta[nodes[0].node_id] = {
            "relevant": True,
            "reason": f"Document root '{nodes[0].title}' — starting point for retrieval.",
            "status": "kept"
        }

    # Phase 1 — top-down pruning
    top = nodes[0].children if (len(nodes) == 1 and nodes[0].children) else nodes
    surviving = _prune_and_collect(
        top, query, index, node_assignment, node_meta, run, retrieval_model
    )
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
                    index.breadcrumb(leaf.node_id),
                    index.parent_summary(leaf.node_id),
                    *node_assignment.get(leaf.node_id, instances[0]),
                    retrieval_model,
                    run=run,
                ): leaf
                for leaf in surviving
            }
            for future in as_completed(futures):
                node_id, meta = future.result()
                node_meta[node_id] = meta
                run.leaf_complete()

    # Preserve original document order
    selected = [l for l in leaves if node_meta.get(l.node_id, {}).get("relevant")]
    return selected, node_meta
