"""
Phase 2 — batched, summary-driven pruning.

The old design made one LLM call per internal node, each seeing only its own
subtree's heading titles. That cost one call per node and gave the model no
basis for preferring one branch over its siblings — with a recall-biased
prompt in isolation, a small model keeps nearly everything.

Now: one call per PARENT, listing its direct children with their index-time
summaries, choosing among them comparatively.
"""
import json

import pytest

import config as app_config
import pageindex

pytestmark = pytest.mark.unit


def _leaf(i, summary=None, content="content"):
    return pageindex.PageNode(node_id=f"leaf-{i}", title=f"Leaf {i}", heading_level=3,
                              line_idx=i, summary=summary or f"Summary of leaf {i}",
                              content=content)


def _section(node_id, children, summary=None):
    return pageindex.PageNode(node_id=node_id, title=node_id.title(), heading_level=2,
                              line_idx=0, summary=summary or f"Summary of {node_id}",
                              children=children)


def _keep(*ids):
    return json.dumps({"keep": [{"id": i, "reason": "looks relevant"} for i in ids]})


def _prune(tree, recorder, response=None):
    """Run the pruning phase over `tree` and return the surviving leaves."""
    if response is not None:
        recorder.default_response = response
    ctx = pageindex.new_run()
    parent_map = pageindex._build_parent_map(tree)
    nodes_by_id = pageindex._build_nodes_by_id(tree)
    assignment = {n.node_id: (None, "") for n in pageindex._collect_all_nodes(tree)}
    meta: dict = {}
    leaves = pageindex._prune_and_collect(
        tree, "the query", parent_map, nodes_by_id, assignment, meta, ctx=ctx)
    return leaves, meta, ctx


class TestBatching:
    def test_one_call_per_parent_not_per_node(self, fake_ollama):
        """6 siblings under one parent used to cost 6 calls."""
        tree = [_section("root", [_leaf(i) for i in range(6)])]
        fake_ollama.default_response = _keep("root")
        _prune(tree, fake_ollama)

        # One call for the root group, one for root's children.
        fake_ollama.assert_count(2, "child-selection calls")

    def test_all_siblings_appear_in_a_single_prompt(self, fake_ollama):
        tree = [_section("root", [_leaf(i) for i in range(6)])]
        fake_ollama.default_response = _keep("root", *[f"leaf-{i}" for i in range(6)])
        _prune(tree, fake_ollama)

        child_prompt = fake_ollama.prompts[1]
        for i in range(6):
            assert f"id=leaf-{i}" in child_prompt, "siblings must be compared together"

    def test_prompt_carries_summaries_not_descendant_content(self, fake_ollama):
        deep = _section("branch", [_leaf(1, content="SECRET-CONTENT-STRING")],
                        summary="Branch summary text")
        tree = [_section("root", [deep])]
        fake_ollama.default_response = _keep("root", "branch")
        _prune(tree, fake_ollama)

        top = fake_ollama.prompts[1]
        assert "Branch summary text" in top
        assert "SECRET-CONTENT-STRING" not in top, (
            "pruning reads summaries; it must not pull descendant content into "
            "the prompt")


class TestSelection:
    def test_unselected_branch_is_pruned_with_its_subtree(self, fake_ollama):
        keep_branch = _section("keep", [_leaf(1)])
        drop_branch = _section("drop", [_leaf(2), _section("deep", [_leaf(3)])])
        tree = [_section("root", [keep_branch, drop_branch])]

        def handler(prompt, model):
            if "id=root" in prompt:
                return _keep("root")
            if "id=keep" in prompt:
                return _keep("keep")
            return _keep("leaf-1")
        fake_ollama.handler = handler

        leaves, meta, ctx = _prune(tree, fake_ollama)
        ids = {l.node_id for l in leaves}
        assert ids == {"leaf-1"}
        for node_id in ("drop", "leaf-2", "deep", "leaf-3"):
            assert meta[node_id]["status"] == "pruned"
            assert node_id in ctx.events_snapshot()["pruned"]

    def test_kept_leaf_is_a_candidate_not_a_verdict(self, fake_ollama):
        """Leaves are nominated here; their real verdict comes from the
        per-leaf evaluator in Phase 3."""
        tree = [_section("root", [_leaf(1)])]
        fake_ollama.handler = lambda p, m: _keep("root") if "id=root" in p else _keep("leaf-1")
        leaves, meta, ctx = _prune(tree, fake_ollama)

        assert [l.node_id for l in leaves] == ["leaf-1"]
        assert "leaf-1" not in meta, "a nominated leaf must carry no verdict yet"
        assert "leaf-1" not in ctx.events_snapshot()["retrieved"]

    def test_progress_reaches_total_when_branches_are_pruned(self, fake_ollama):
        """Regression guard: pruned leaves must still count as done, or the
        progress bar never fills."""
        tree = [_section("root", [_section("drop", [_leaf(1), _leaf(2)])])]
        fake_ollama.handler = lambda p, m: _keep("root") if "id=root" in p else _keep()

        ctx = pageindex.new_run(total_leaves=2)
        parent_map = pageindex._build_parent_map(tree)
        nodes_by_id = pageindex._build_nodes_by_id(tree)
        assignment = {n.node_id: (None, "") for n in pageindex._collect_all_nodes(tree)}
        pageindex._prune_and_collect(tree, "q", parent_map, nodes_by_id, assignment, {}, ctx=ctx)

        assert ctx.progress() == {"total": 2, "done": 2}


class TestRobustness:
    def test_unknown_ids_are_ignored(self, fake_ollama, capsys):
        tree = [_section("root", [_leaf(1)])]
        fake_ollama.handler = lambda p, m: (
            _keep("root") if "id=root" in p else _keep("hallucinated-id"))
        leaves, meta, _ = _prune(tree, fake_ollama)

        assert leaves == []
        assert "hallucinated-id" not in meta
        assert "ignoring unknown id" in capsys.readouterr().err

    def test_empty_top_level_selection_prunes_the_document(self, fake_ollama):
        """"This document has nothing to do with the question" is a legitimate
        verdict — it is exactly what cross-document retrieval needs. Overriding
        it forced every document to be explored in full for every query, and
        (worse) reported the override as a failure in the UI."""
        tree = [_section("a", [_leaf(1)]), _section("b", [_leaf(2)])]
        fake_ollama.default_response = _keep()
        leaves, meta, ctx = _prune(tree, fake_ollama)

        assert leaves == [], "an irrelevant document must prune away entirely"
        assert meta["a"]["status"] == "pruned"
        assert ctx.events_snapshot()["error"] == [], (
            "a decision we disagree with is not an error")
        assert "Not relevant to this question" in meta["a"]["reason"]

    def test_deeper_empty_selection_does_prune(self, fake_ollama):
        """Below the root, selecting nothing is a legitimate verdict."""
        tree = [_section("root", [_section("branch", [_leaf(1)])])]
        fake_ollama.handler = lambda p, m: _keep("root") if "id=root" in p else _keep()
        leaves, meta, _ = _prune(tree, fake_ollama)
        assert leaves == []
        assert meta["branch"]["status"] == "pruned"

    def test_malformed_response_keeps_the_group_and_flags_error(self, fake_ollama):
        tree = [_section("root", [_section("branch", [_leaf(1)])])]
        fake_ollama.handler = lambda p, m: (
            _keep("root") if "id=root" in p else "not json at all")
        leaves, meta, ctx = _prune(tree, fake_ollama)

        assert meta["branch"]["relevant"] is True, "never prune on a failed call"
        assert meta["branch"]["status"] == "error", (
            "a failure must be distinguishable from a decision")
        assert "branch" in ctx.events_snapshot()["error"]

    def test_bare_list_response_is_accepted(self, fake_ollama):
        tree = [_section("root", [_leaf(1)])]
        fake_ollama.handler = lambda p, m: (
            json.dumps(["root"]) if "id=root" in p else json.dumps(["leaf-1"]))
        leaves, _, _ = _prune(tree, fake_ollama)
        assert [l.node_id for l in leaves] == ["leaf-1"]


class TestNodeIdCleaning:
    """`_clean_node_id` stripped a leading LEAF/SECTION with a zero-length
    separator, so it also ate the prefix of any legitimate id that merely
    started with those words."""

    @pytest.mark.parametrize("raw,expected", [
        ("leaf-1", "leaf-1"),
        ("section-overview", "section-overview"),
        ("sections-of-the-heart", "sections-of-the-heart"),
        ("leaflet-anatomy", "leaflet-anatomy"),
        # The prefixes the cleaner genuinely exists to strip:
        ("LEAF: node-abc", "node-abc"),
        ("SECTION  node-abc", "node-abc"),
        ("LEAF | id=node-abc | Title", "node-abc"),
        ("id=node-abc", "node-abc"),
    ])
    def test_ids_survive_cleaning(self, raw, expected):
        assert pageindex._clean_node_id(raw) == expected

    def test_a_document_heading_named_section_still_prunes(self, fake_ollama):
        """End-to-end version: a real '## Section overview' heading slugs to
        'section-overview', which the old cleaner turned into '-overview'."""
        child = pageindex.PageNode(node_id="section-overview", title="Section overview",
                                   heading_level=2, line_idx=1, summary="s", content="c")
        tree = [_section("root", [child])]
        fake_ollama.handler = lambda p, m: (
            _keep("root") if "id=root" in p else _keep("section-overview"))
        leaves, _, _ = _prune(tree, fake_ollama)
        assert [l.node_id for l in leaves] == ["section-overview"]


class TestChunking:
    def test_wide_fanout_splits_into_batches(self, fake_ollama):
        app_config.update(retrieval_ctx=4096)
        children = [_leaf(i, summary="s" * 400) for i in range(40)]
        tree = [_section("root", children)]
        fake_ollama.default_response = _keep("root")
        _prune(tree, fake_ollama)

        # 1 root call + more than one call for the oversized child list.
        assert fake_ollama.count > 2, "a 40-child list must not go out as one prompt"

    def test_batches_cover_every_child_exactly_once(self, fake_ollama):
        app_config.update(retrieval_ctx=4096)
        children = [_leaf(i, summary="s" * 400) for i in range(40)]
        batches = pageindex._child_batches(children, "crumb", "query")
        seen = [c.node_id for b in batches for c in b]
        assert len(batches) > 1
        assert seen == [c.node_id for c in children]

    def test_small_fanout_is_a_single_batch(self, fake_ollama):
        children = [_leaf(i) for i in range(8)]
        assert len(pageindex._child_batches(children, "crumb", "query")) == 1

    def test_union_across_batches(self, fake_ollama):
        app_config.update(retrieval_ctx=4096)
        children = [_leaf(i, summary="s" * 400) for i in range(40)]
        tree = [_section("root", children)]

        def handler(prompt, model):
            if "id=root" in prompt and "id=leaf-0" not in prompt:
                return _keep("root")
            # keep the first child mentioned in whichever batch this is
            first = prompt.split("- id=")[1].split(" ")[0]
            return _keep(first)
        fake_ollama.handler = handler

        leaves, _, _ = _prune(tree, fake_ollama)
        assert len(leaves) == len(pageindex._child_batches(children, "", "")), (
            "one survivor per batch — results must be unioned, not overwritten")
