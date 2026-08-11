"""
Phase 3 — context-budgeted leaf evaluation.

Every retrieved node ends up in the AGENT's context, so that window is the
real constraint, and the cap is on ACCEPTED nodes. Candidates are evaluated in
BM25 order until the accepted passages would overflow it; rejected leaves cost
nothing and the walk continues past them.

What is left over is `deferred` — never evaluated, budget exhausted. That is
not a judgement and must never be conflated with `pruned` or `rejected`.
"""
import json

import pytest

import config as app_config
import pageindex

pytestmark = pytest.mark.unit


def _leaf(i, content):
    return pageindex.PageNode(node_id=f"leaf-{i}", title=f"Leaf {i}", heading_level=2,
                              line_idx=i, summary=f"Summary {i}", content=content)


def _doc(leaves, name="doc"):
    root = pageindex.PageNode(node_id="root", title="Root", heading_level=1,
                              line_idx=0, summary="root", children=leaves)
    return pageindex.DocCandidates(
        doc_name=name, nodes=[root], leaves=leaves,
        parent_map=pageindex._build_parent_map([root]),
        nodes_by_id=pageindex._build_nodes_by_id([root]),
        node_assignment={n.node_id: (None, "")
                         for n in pageindex._collect_all_nodes([root])},
        candidates=list(leaves), node_meta={})


RELEVANT = '{"relevant": true, "reason": "answers it", "quote": "q"}'
IRRELEVANT = '{"relevant": false, "reason": "off topic"}'


class TestBudgetCutoff:
    def test_everything_fits_nothing_deferred(self, fake_ollama):
        fake_ollama.default_response = RELEVANT
        doc = _doc([_leaf(i, "short passage") for i in range(5)])
        ctx = pageindex.new_run()

        pageindex.evaluate_ranked([doc], "query", ctx=ctx)

        assert ctx.budget.deferred == 0
        assert ctx.budget.capped_by == ""
        assert len(ctx.events_snapshot()["retrieved"]) == 5
        assert ctx.events_snapshot()["deferred"] == []

    def test_budget_exhaustion_defers_the_tail(self, fake_ollama):
        app_config.update(agent_ctx=4096)
        fake_ollama.default_response = RELEVANT
        doc = _doc([_leaf(i, "word " * 2000) for i in range(10)])
        ctx = pageindex.new_run()

        pageindex.evaluate_ranked([doc], "query", ctx=ctx)

        assert ctx.budget.capped_by == "context"
        assert ctx.budget.deferred > 0
        assert ctx.budget.tokens_used <= ctx.budget.tokens_max

    def test_rejected_leaves_do_not_consume_budget(self, fake_ollama):
        """A rejected passage never reaches the agent, so it must not use up
        the agent's window — the walk continues past it."""
        app_config.update(agent_ctx=8192)
        big = "word " * 3000

        def handler(prompt, model):
            return RELEVANT if "leaf-9" in prompt else IRRELEVANT
        fake_ollama.handler = handler

        doc = _doc([_leaf(i, big) for i in range(10)])
        ctx = pageindex.new_run()
        pageindex.evaluate_ranked([doc], "query", ctx=ctx)

        assert ctx.budget.evaluated == 10, (
            "nine rejections must not stop the walk before the tenth leaf")
        assert ctx.budget.deferred == 0

    def test_nothing_fits_is_graceful(self, fake_ollama):
        app_config.update(agent_ctx=4096)
        fake_ollama.default_response = RELEVANT
        doc = _doc([_leaf(i, "word " * 50000) for i in range(3)])
        ctx = pageindex.new_run()

        pageindex.evaluate_ranked([doc], "query", ctx=ctx)

        assert ctx.events_snapshot()["retrieved"] == []
        assert ctx.budget.deferred == 3

    def test_no_candidates_is_a_noop(self, fake_ollama):
        doc = _doc([])
        ctx = pageindex.new_run()
        pageindex.evaluate_ranked([doc], "query", ctx=ctx)
        fake_ollama.assert_count(0)

    def test_max_evals_cap_is_reported_separately(self, fake_ollama):
        app_config.update(max_leaf_evals=3, agent_ctx=131072)
        fake_ollama.default_response = IRRELEVANT
        doc = _doc([_leaf(i, "short") for i in range(20)])
        ctx = pageindex.new_run()

        pageindex.evaluate_ranked([doc], "query", ctx=ctx)

        assert ctx.budget.capped_by == "max_evals"
        assert ctx.budget.evaluated == 3
        assert ctx.budget.deferred == 17


class TestDeferredSemantics:
    def test_deferred_is_distinct_from_pruned_and_rejected(self, fake_ollama):
        app_config.update(agent_ctx=4096)
        fake_ollama.default_response = RELEVANT
        doc = _doc([_leaf(i, "word " * 2000) for i in range(10)])
        ctx = pageindex.new_run()
        pageindex.evaluate_ranked([doc], "query", ctx=ctx)

        snap = ctx.events_snapshot()
        assert snap["deferred"], "some leaves should be deferred"
        assert snap["pruned"] == [], "deferral is not pruning"
        assert set(snap["deferred"]).isdisjoint(snap["rejected"])
        assert set(snap["deferred"]).isdisjoint(snap["retrieved"])

    def test_deferred_nodes_carry_their_rank(self, fake_ollama):
        app_config.update(agent_ctx=4096)
        fake_ollama.default_response = RELEVANT
        doc = _doc([_leaf(i, "word " * 2000) for i in range(10)])
        ctx = pageindex.new_run()
        pageindex.evaluate_ranked([doc], "query", ctx=ctx)

        for node_id in ctx.events_snapshot()["deferred"]:
            meta = ctx.get_meta(node_id)
            assert meta.get("bm25_rank") is not None, (
                "a deferred node must say where it sat in the ranking, so the "
                "user can judge what they are not seeing")

    def test_deferred_reason_explains_it_was_not_a_judgement(self, fake_ollama):
        app_config.update(agent_ctx=4096)
        fake_ollama.default_response = RELEVANT
        doc = _doc([_leaf(i, "word " * 2000) for i in range(10)])
        ctx = pageindex.new_run()
        pageindex.evaluate_ranked([doc], "query", ctx=ctx)

        reasons = [ctx.get_meta(n)["reason"] for n in ctx.events_snapshot()["deferred"]]
        assert all("context budget" in r or "evaluation cap" in r for r in reasons)


class TestAccountingInvariant:
    """The bug class this design is most prone to: leaves quietly vanishing
    between the phases, or being counted twice."""

    @pytest.mark.parametrize("agent_ctx,n_leaves,size", [
        (131072, 12, "short"),
        (4096, 12, "word " * 2000),
        (8192, 30, "word " * 400),
    ])
    def test_every_candidate_ends_in_exactly_one_state(
            self, fake_ollama, agent_ctx, n_leaves, size):
        app_config.update(agent_ctx=agent_ctx)
        fake_ollama.default_response = RELEVANT
        leaves = [_leaf(i, size) for i in range(n_leaves)]
        doc = _doc(leaves)
        ctx = pageindex.new_run(total_leaves=n_leaves)

        pageindex.evaluate_ranked([doc], "query", ctx=ctx)

        snap = ctx.events_snapshot()
        buckets = ["retrieved", "rejected", "deferred", "error"]
        seen = [nid for b in buckets for nid in snap[b]]
        assert len(seen) == len(set(seen)), "a leaf may not occupy two states"
        assert set(seen) == {l.node_id for l in leaves}, (
            "every candidate must end in exactly one terminal state")

    def test_progress_reaches_total(self, fake_ollama):
        app_config.update(agent_ctx=4096)
        fake_ollama.default_response = RELEVANT
        leaves = [_leaf(i, "word " * 2000) for i in range(10)]
        ctx = pageindex.new_run(total_leaves=10)
        pageindex.evaluate_ranked([_doc(leaves)], "query", ctx=ctx)
        assert ctx.progress() == {"total": 10, "done": 10}

    def test_node_meta_covers_every_candidate(self, fake_ollama):
        app_config.update(agent_ctx=4096)
        fake_ollama.default_response = RELEVANT
        leaves = [_leaf(i, "word " * 2000) for i in range(10)]
        doc = _doc(leaves)
        pageindex.evaluate_ranked([doc], "query", ctx=pageindex.new_run())
        assert {l.node_id for l in leaves} <= set(doc.node_meta)


class TestOrdering:
    def test_evaluation_follows_the_ranking(self, fake_ollama):
        fake_ollama.default_response = IRRELEVANT
        leaves = [
            _leaf(0, "unrelated committee minutes"),
            _leaf(1, "sodium restriction lowers blood pressure"),
            _leaf(2, "more unrelated filler text"),
        ]
        pageindex.evaluate_ranked([_doc(leaves)], "sodium blood pressure",
                                  ctx=pageindex.new_run())
        assert "sodium restriction" in fake_ollama.prompts[0], (
            "the best-ranked candidate must be evaluated first")

    def test_cutoff_is_deterministic_under_racing_completions(self, fake_ollama):
        """Results are committed in rank order, so which call finishes first
        must not change which leaves get deferred."""
        import random
        import time

        app_config.update(agent_ctx=6144)

        def handler(prompt, model):
            time.sleep(random.uniform(0, 0.004))
            return RELEVANT
        fake_ollama.handler = handler

        outcomes = []
        for _ in range(4):
            leaves = [_leaf(i, "word " * 900) for i in range(12)]
            ctx = pageindex.new_run()
            pageindex.evaluate_ranked([_doc(leaves)], "query", ctx=ctx)
            outcomes.append(tuple(sorted(ctx.events_snapshot()["deferred"])))

        assert len(set(outcomes)) == 1, f"non-deterministic cutoff: {set(outcomes)}"


class TestMultiDocument:
    def test_budget_and_ranking_span_the_whole_corpus(self, fake_ollama):
        """Every passage lands in one agent context, so the budget cannot be
        applied per document."""
        app_config.update(agent_ctx=4096)
        fake_ollama.default_response = RELEVANT
        docs = [
            _doc([_leaf(i, "word " * 1500) for i in range(5)], name="doc-a"),
            _doc([_leaf(i + 100, "word " * 1500) for i in range(5)], name="doc-b"),
        ]
        ctx = pageindex.new_run()
        pageindex.evaluate_ranked(docs, "query", ctx=ctx)

        accepted = len(ctx.events_snapshot()["retrieved"])
        assert accepted < 10, "the corpus-wide budget must bind across documents"
        assert ctx.budget.tokens_used <= ctx.budget.tokens_max


class TestAgentBudget:
    def test_budget_tracks_the_agent_context_setting(self):
        app_config.update(agent_ctx=32768)
        small = pageindex.agent_budget_tokens("query")
        app_config.update(agent_ctx=65536)
        assert pageindex.agent_budget_tokens("query") > small

    def test_budget_reserves_room_for_the_response(self):
        app_config.update(agent_ctx=32768)
        assert pageindex.agent_budget_tokens("query") < 32768

    def test_a_long_query_reduces_the_budget(self):
        app_config.update(agent_ctx=32768)
        short = pageindex.agent_budget_tokens("q")
        assert pageindex.agent_budget_tokens("word " * 5000) < short
