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
from dataclasses import dataclass
from typing import Optional

import ollama

from .question_run import PassageDecision, PassageDecisionKind, QuestionRun
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

@dataclass(frozen=True)
class DocumentRetrieval:
    """One document's complete typed retrieval audit."""

    decisions: tuple[PassageDecision, ...]

    @property
    def retrieved_node_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.node_id
            for decision in self.decisions
            if decision.kind is PassageDecisionKind.PASSAGE_RETRIEVED
        )


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
) -> PassageDecision:
    """Evaluate one leaf and atomically record its truthful decision."""
    prompt = (
        LEAF_EVAL_PROMPT
        .replace("QUERY_PLACEHOLDER", query)
        .replace("DOCNAME_PLACEHOLDER", doc_name.replace("_", " ").title())
        .replace("BREADCRUMB_PLACEHOLDER", breadcrumb)
        .replace("PARENT_SUMMARY_PLACEHOLDER", parent_summary or "top-level section")
        .replace("CONTENT_PLACEHOLDER", leaf.content or "(no content)")
    )
    failure_code = "evaluation_error"
    decision: PassageDecision | None = None
    failure = "the model returned no usable verdict"
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
                    decision = PassageDecision(
                        leaf.node_id,
                        PassageDecisionKind.PASSAGE_RETRIEVED,
                        reason,
                        quote,
                        document_id=doc_name,
                    )
            else:
                decision = PassageDecision(
                    leaf.node_id,
                    PassageDecisionKind.PASSAGE_REJECTED,
                    reason,
                    document_id=doc_name,
                )
    except Exception as exc:
        print(f"    [warn] leaf eval failed for {leaf.node_id}: {exc}", file=sys.stderr)
        failure = str(exc)
    finally:
        release(client_url)

    if decision is not None:
        run.record_decision(decision)
        return decision

    # Reached only when the model could not be consulted, or answered with
    # something unparseable. That is not a judgement that the passage is
    # irrelevant, and must not be recorded as one: a caller seeing every leaf
    # "rejected" would report "nothing relevant was found" to a clinician when
    # the truth is that nothing was actually checked. Mirrors the section
    # check, which already refuses to prune on error.
    decision = PassageDecision(
        leaf.node_id,
        PassageDecisionKind.PASSAGE_CHECK_FAILED,
        failure,
        code=failure_code,
        document_id=doc_name,
    )
    run.record_decision(decision)
    return decision


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
    document_id: str = "",
) -> PassageDecision:
    """Lightweight LLM call: can this section contain a direct answer?"""
    query_terms = _terms(query)
    direct_matches = sorted(query_terms & _heading_terms(node, index))
    if direct_matches:
        reason = (
            "Kept because the query directly matches descendant heading term(s): "
            + ", ".join(direct_matches)
        )
        decision = PassageDecision(
            node.node_id, PassageDecisionKind.SECTION_KEPT, reason,
            document_id=document_id,
        )
        run.record_decision(decision)
        print(
            f"    [prune-check] {node.node_id}: KEEP (heading match: "
            f"{', '.join(direct_matches)})",
            file=sys.stderr,
        )
        return decision

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
            decision = PassageDecision(
                node.node_id, PassageDecisionKind.SECTION_KEPT, reason,
                document_id=document_id,
            )
        else:
            decision = PassageDecision(
                node.node_id, PassageDecisionKind.SECTION_PRUNED, reason,
                document_id=document_id,
            )
        run.record_decision(decision)

        print(f"    [prune-check] {node.node_id}: {'KEEP' if verdict else 'PRUNE'} (raw={raw!r:.120})", file=sys.stderr)
        return decision
    except Exception as exc:
        decision = PassageDecision(
            node.node_id,
            PassageDecisionKind.SECTION_CHECK_FAILED,
            f"Error checking section: {exc}",
            code="section_check_failed",
            document_id=document_id,
        )
        run.record_decision(decision)
        print(f"    [prune-check] {node.node_id}: ERROR ({exc}) — keeping", file=sys.stderr)
        return decision  # conservative: never prune on error
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
    decisions: dict[str, PassageDecision],
    run: QuestionRun,
    model: str | None = None,
    document_id: str = "",
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
                    document_id=document_id,
                ): node
                for node in sections
            }
            next_frontier: list["PageNode"] = []
            for future in as_completed(futures):
                node = futures[future]
                decision = future.result()
                decisions[node.node_id] = decision

                if decision.kind.relevant:
                    next_frontier.extend(node.children)
                else:
                    # Pruned: count all leaf descendants toward "done" so the
                    # progress counter reaches total. They won't be evaluated again.
                    _prune_reason = f"Pruned because ancestor section '{node.title}' was pruned."
                    for descendant in index.descendants(
                        node.node_id
                    ):
                        descendant_decision = PassageDecision(
                            descendant.node_id,
                            (
                                PassageDecisionKind.PASSAGE_PRUNED
                                if descendant.is_leaf
                                else PassageDecisionKind.SECTION_PRUNED
                            ),
                            _prune_reason,
                            document_id=document_id,
                        )
                        run.record_decision(descendant_decision)
                        decisions[descendant.node_id] = descendant_decision

        frontier = next_frontier

    return candidate_leaves


def search_document(
    doc_name: str,
    query: str,
    run: Optional[QuestionRun],
    index: DocumentIndex,
    model: str | None = None,
    instances=None,
) -> DocumentRetrieval:
    """Two-phase retrieval with top-down pruning.

    Phase 1: BFS section checks — prune branches whose sections are irrelevant.
    Phase 2: full leaf evaluation (reason + quote) on the survivors only.

    Records into `run` as it goes, so a client can watch. Without one, the
    verdicts are still returned; only the live view is skipped.

    Returns typed passage decisions in document order. Selected evidence is
    derived from retrieved decisions by the Whole-library search module.
    """
    run = run or QuestionRun()
    retrieval_model = model or settings.model
    nodes = list(index.nodes)
    leaves = list(index.leaves)

    if not leaves:
        return DocumentRetrieval(())
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

    decisions: dict[str, PassageDecision] = {}
    if len(nodes) == 1 and nodes[0].children:
        root = nodes[0]
        root_decision = PassageDecision(
            root.node_id,
            PassageDecisionKind.SECTION_KEPT,
            f"Document root '{root.title}' — starting point for retrieval.",
            document_id=doc_name,
        )
        decisions[root.node_id] = root_decision
        run.record_decision(root_decision)

    # Phase 1 — top-down pruning
    top = nodes[0].children if (len(nodes) == 1 and nodes[0].children) else nodes
    surviving = _prune_and_collect(
        top, query, index, node_assignment, decisions, run, retrieval_model,
        document_id=doc_name,
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
                decision = future.result()
                decisions[decision.node_id] = decision

    # Preserve original document order
    order = index.node_order()
    ordered_decisions = tuple(sorted(
        decisions.values(), key=lambda item: order.get(item.node_id, len(order))
    ))
    return DocumentRetrieval(ordered_decisions)
