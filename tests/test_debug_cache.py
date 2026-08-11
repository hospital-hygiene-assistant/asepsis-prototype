"""
Phase 6 — debug cache.

Exact-key replay of a previous run. The key includes an index fingerprint,
which is the part that actually matters: without it, re-ingesting would keep
replaying stale results and you would debug a "regression" that is really a
cache hit from before the change.
"""
import json
import time

import pytest

import config as app_config
import debug_cache as dc
import pageindex

pytestmark = pytest.mark.unit


@pytest.fixture
def cache(tmp_path):
    app_config.update(debug_cache_enabled=True)
    return dc.DebugCache(tmp_path / "cache")


@pytest.fixture
def index_dir(tmp_path):
    d = tmp_path / "index"
    d.mkdir()
    (d / "doc.json").write_text(json.dumps({"formatVersion": 2, "nodes": []}))
    return d


RESULTS = {"doc": {"tree": [], "retrieved_ids": ["a", "b"], "node_meta": {}, "nodes": []}}
EVENTS = {"retrieved": ["a", "b"], "pruned": ["c"], "deferred": ["d"],
          "kept": [], "rejected": [], "error": [],
          "meta": {"a": {"status": "retrieved", "reason": "why", "quote": "q"}},
          "budget": {"tokens_used": 100, "deferred": 1}}
BUDGET = {"evaluated": 3, "deferred": 1}


class TestKey:
    def _key(self, index_dir, **kw):
        return dc.make_key("the query", index_dir, **kw)

    def test_identical_inputs_give_the_same_key(self, index_dir):
        assert self._key(index_dir) == self._key(index_dir)

    def test_query_is_normalised(self, index_dir):
        assert dc.make_key("  The   Query  ", index_dir) == dc.make_key("the query", index_dir)

    def test_different_query_different_key(self, index_dir):
        assert self._key(index_dir) != dc.make_key("another query", index_dir)

    def test_index_change_changes_the_key(self, index_dir):
        before = self._key(index_dir)
        (index_dir / "doc.json").write_text(json.dumps(
            {"formatVersion": 2, "nodes": [{"nodeId": "x"}]}))
        assert self._key(index_dir) != before, (
            "a re-index MUST invalidate — stale replay is the trap this exists to avoid")

    def test_adding_a_document_changes_the_key(self, index_dir):
        before = self._key(index_dir)
        (index_dir / "doc2.json").write_text(json.dumps({"formatVersion": 2, "nodes": []}))
        assert self._key(index_dir) != before

    def test_tags_change_the_key(self, index_dir):
        assert self._key(index_dir, tags=["arxiv"]) != self._key(index_dir)

    def test_tag_order_does_not_change_the_key(self, index_dir):
        assert (self._key(index_dir, tags=["a", "b"]) ==
                self._key(index_dir, tags=["b", "a"]))

    def test_selected_answers_change_the_key(self, index_dir):
        assert self._key(index_dir, selected_answers=["q1__icu"]) != self._key(index_dir)

    def test_model_changes_the_key(self, index_dir):
        assert self._key(index_dir, retrieval_model="m2") != self._key(index_dir)

    def test_prompt_version_changes_the_key(self, index_dir, monkeypatch):
        before = self._key(index_dir)
        bumped = dict(app_config.PROMPT_VERSIONS)
        bumped["leaf_eval"] += 1
        monkeypatch.setattr(app_config, "PROMPT_VERSIONS", bumped)
        assert self._key(index_dir) != before

    def test_agent_context_changes_the_key(self, index_dir):
        before = self._key(index_dir)
        app_config.update(agent_ctx=32768)
        assert self._key(index_dir) != before, (
            "the context window decides what gets deferred, so it is part of the run")

    def test_missing_index_dir_is_handled(self, tmp_path):
        assert dc.make_key("q", tmp_path / "nope")


class TestRoundTrip:
    def test_store_and_retrieve(self, cache):
        cache.put("k1", query="q", results=RESULTS, events=EVENTS, budget=BUDGET)
        entry = cache.get("k1")
        assert entry is not None
        assert entry.results == RESULTS
        assert entry.events == EVENTS
        assert entry.budget == BUDGET

    def test_miss_returns_none(self, cache):
        assert cache.get("nothing-here") is None

    def test_summary_reports_node_count(self, cache):
        cache.put("k1", query="q", results=RESULTS, events=EVENTS, budget=BUDGET)
        assert cache.get("k1").summary()["node_count"] == 2

    def test_corrupt_entry_is_a_miss_not_a_crash(self, cache):
        cache.dir.mkdir(parents=True, exist_ok=True)
        (cache.dir / "bad.json").write_text("{ not json")
        assert cache.get("bad") is None

    def test_answer_is_stored_separately(self, cache):
        cache.put("k1", query="q", results=RESULTS, events=EVENTS, budget=BUDGET,
                  answer={"content": "the answer"})
        assert cache.get("k1").answer["content"] == "the answer"


class TestDisabled:
    def test_disabled_never_writes(self, tmp_path):
        app_config.update(debug_cache_enabled=False)
        c = dc.DebugCache(tmp_path / "cache")
        assert c.put("k", query="q", results={}, events={}, budget={}) is None
        assert not (tmp_path / "cache").exists()

    def test_disabled_always_misses(self, tmp_path):
        app_config.update(debug_cache_enabled=True)
        c = dc.DebugCache(tmp_path / "cache")
        c.put("k", query="q", results=RESULTS, events=EVENTS, budget=BUDGET)
        app_config.update(debug_cache_enabled=False)
        assert c.get("k") is None

    def test_disabled_by_default(self):
        app_config.reset_for_tests()
        assert app_config.runtime().debug_cache_enabled is False, (
            "a debug aid must be opt-in")


class TestReplayFidelity:
    def test_restoring_events_repopulates_the_visualisation(self, cache):
        """Regression guard: starting a run clears every event set, so without
        an explicit restore a cache hit renders an empty treemap."""
        cache.put("k1", query="q", results=RESULTS, events=EVENTS, budget=BUDGET)
        entry = cache.get("k1")

        ctx = pageindex.new_run(total_leaves=2)
        ctx.restore(entry.events)

        snap = ctx.events_snapshot()
        assert snap["retrieved"] == ["a", "b"]
        assert snap["pruned"] == ["c"]
        assert snap["deferred"] == ["d"], "deferred state must survive a replay too"
        assert ctx.get_meta("a")["quote"] == "q"

    def test_replayed_results_are_identical(self, cache):
        cache.put("k1", query="q", results=RESULTS, events=EVENTS, budget=BUDGET)
        assert cache.get("k1").results == RESULTS


class TestQuestionListing:
    def test_lists_newest_first(self, cache, index_dir):
        for i, q in enumerate(["first", "second", "third"]):
            key = dc.make_key(q, index_dir)
            cache.put(key, query=q, results=RESULTS, events=EVENTS, budget=BUDGET,
                      index_dir=index_dir)
            time.sleep(0.01)
        assert [e.query for e in cache.entries(index_dir)] == ["third", "second", "first"]

    def test_stale_entries_are_not_offered(self, cache, index_dir):
        """A question whose key no longer matches the corpus must never appear
        in autocomplete — offering it would promise a replay that is wrong."""
        key = dc.make_key("a question", index_dir)
        cache.put(key, query="a question", results=RESULTS, events=EVENTS,
                  budget=BUDGET, index_dir=index_dir)
        assert len(cache.entries(index_dir)) == 1

        (index_dir / "doc.json").write_text(json.dumps(
            {"formatVersion": 2, "nodes": [{"nodeId": "changed"}]}))
        assert cache.entries(index_dir) == []

    def test_listing_without_an_index_dir_returns_everything(self, cache, index_dir):
        cache.put("k1", query="q", results=RESULTS, events=EVENTS, budget=BUDGET)
        assert len(cache.entries()) == 1

    def test_clear_empties_the_cache(self, cache, index_dir):
        for q in ("a", "b"):
            cache.put(dc.make_key(q, index_dir), query=q, results=RESULTS,
                      events=EVENTS, budget=BUDGET, index_dir=index_dir)
        assert cache.clear() == 2
        assert cache.entries() == []


class TestFingerprint:
    def test_stable_across_reads(self, index_dir):
        assert dc.index_fingerprint(index_dir) == dc.index_fingerprint(index_dir)

    def test_content_not_mtime(self, index_dir):
        """A no-op rebuild must not invalidate; a same-second edit must."""
        before = dc.index_fingerprint(index_dir)
        p = index_dir / "doc.json"
        p.write_text(p.read_text())          # rewritten, identical content
        assert dc.index_fingerprint(index_dir) == before

        p.write_text(json.dumps({"formatVersion": 2, "nodes": [{"nodeId": "z"}]}))
        assert dc.index_fingerprint(index_dir) != before

    def test_missing_dir(self, tmp_path):
        assert dc.index_fingerprint(tmp_path / "nope") == "empty"
