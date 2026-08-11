"""
Phase 0 — per-run state isolation and the bounded worker pool.

Before RunContext, run state lived in module-level globals, so two concurrent
retrievals overwrote each other's verdicts and progress. That is the bug these
tests pin down — it becomes load-bearing in Phase 5, where a background
precompute runs alongside a live user query.
"""
import threading

import pytest

import config as app_config
import pageindex

pytestmark = pytest.mark.unit


class TestIsolation:
    def test_two_runs_do_not_see_each_others_events(self):
        a = pageindex.new_run(total_leaves=10, make_current=True)
        b = pageindex.new_run(total_leaves=5, make_current=False)

        a.mark("node-1", "retrieved", "found it")
        b.mark("node-2", "pruned", "not here")

        assert a.events_snapshot()["retrieved"] == ["node-1"]
        assert a.events_snapshot()["pruned"] == []
        assert b.events_snapshot()["pruned"] == ["node-2"]
        assert b.events_snapshot()["retrieved"] == []

    def test_background_run_does_not_steal_current(self):
        live = pageindex.new_run(total_leaves=10, make_current=True)
        pageindex.new_run(total_leaves=99, make_current=False)
        assert pageindex.current_run().run_id == live.run_id
        assert pageindex.get_progress()["total"] == 10

    def test_progress_is_per_run(self):
        a = pageindex.new_run(total_leaves=10, make_current=True)
        b = pageindex.new_run(total_leaves=10, make_current=False)
        a.inc_done()
        a.inc_done()
        b.inc_done()
        assert a.progress()["done"] == 2
        assert b.progress()["done"] == 1

    def test_concurrent_marking_is_threadsafe(self):
        ctx = pageindex.new_run(total_leaves=0)

        def worker(base):
            for i in range(200):
                ctx.mark(f"n{base}-{i}", "retrieved")
                ctx.inc_done()

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ctx.progress()["done"] == 1600
        assert len(ctx.events_snapshot()["retrieved"]) == 1600


class TestStatusTransitions:
    def test_a_node_holds_only_one_status(self):
        ctx = pageindex.new_run()
        ctx.mark("n1", "kept", "maybe")
        ctx.mark("n1", "pruned", "actually no")
        snap = ctx.events_snapshot()
        assert snap["pruned"] == ["n1"]
        assert snap["kept"] == [], "stale status must not linger in the old bucket"
        assert ctx.get_meta("n1")["status"] == "pruned"

    def test_all_statuses_present_in_snapshot(self):
        snap = pageindex.new_run().events_snapshot()
        for status in pageindex.STATUSES:
            assert status in snap
        assert "deferred" in snap, "Phase 3 needs deferred as a first-class status"

    def test_snapshot_is_a_copy(self):
        ctx = pageindex.new_run()
        ctx.mark("n1", "kept")
        snap = ctx.events_snapshot()
        snap["kept"].append("injected")
        snap["meta"]["n1"]["status"] = "tampered"
        assert ctx.events_snapshot()["kept"] == ["n1"]
        assert ctx.get_meta("n1")["status"] == "kept"


class TestRestore:
    def test_roundtrip(self):
        """The debug cache replays a run through restore(); if this loses
        events the treemap comes up empty on a cache hit."""
        src = pageindex.new_run(total_leaves=3)
        src.mark("a", "retrieved", "why", "quote")
        src.mark("b", "pruned", "nope")
        src.mark("c", "deferred", "budget")
        src.budget.tokens_used = 1234
        src.budget.deferred = 1

        dst = pageindex.new_run()
        dst.restore(src.events_snapshot())

        assert dst.events_snapshot()["retrieved"] == ["a"]
        assert dst.events_snapshot()["deferred"] == ["c"]
        assert dst.get_meta("a")["quote"] == "quote"
        assert dst.budget.tokens_used == 1234
        assert dst.budget.deferred == 1

    def test_restore_clears_prior_state(self):
        ctx = pageindex.new_run()
        ctx.mark("stale", "retrieved")
        ctx.restore({"retrieved": ["fresh"], "meta": {}})
        assert ctx.events_snapshot()["retrieved"] == ["fresh"]


class TestWorkPool:
    def test_size_follows_instances_and_concurrency(self):
        pageindex.reconfigure_clients(["http://a:1", "http://b:2"])
        app_config.update(concurrency_per_instance=3)
        assert pageindex.pool_size() == 6

    def test_pool_is_bounded_under_large_fanout(self):
        """The old code created one thread per node — hundreds of threads all
        queueing on two Ollama instances."""
        pageindex.reconfigure_clients(["http://a:1"])
        app_config.update(concurrency_per_instance=2)
        pageindex._reset_work_pool()

        pool = pageindex.work_pool()
        seen: set[str] = set()
        lock = threading.Lock()

        def task(_):
            with lock:
                seen.add(threading.current_thread().name)

        list(pool.map(task, range(500)))
        assert len(seen) <= 2, f"pool leaked threads: {len(seen)}"

    def test_pool_is_reused_across_calls(self):
        assert pageindex.work_pool() is pageindex.work_pool()

    def test_reconfigure_rebuilds_the_pool(self):
        first = pageindex.work_pool()
        pageindex.reconfigure_clients(["http://a:1", "http://b:2"])
        assert pageindex.work_pool() is not first


class TestPoolNesting:
    """Orchestration and LLM work must not share a pool.

    Per-document retrieval tasks block on the node-level calls they submit.
    Running both on one bounded pool deadlocks it: N documents occupy all N
    workers, and each then waits for a worker that can never free up. With 2
    workers and 5 documents, every query hung forever with no progress and no
    error — the failure mode is indistinguishable from "the model is slow".
    """

    def test_pools_are_distinct(self):
        assert pageindex.work_pool() is not pageindex.coordination_pool()

    def test_coordination_pool_is_large_enough_to_never_be_the_bottleneck(self):
        assert pageindex.coordination_pool()._max_workers >= 16

    def test_nested_submission_across_pools_completes(self):
        """The exact shape of _run_retrieval: more documents than workers,
        each submitting node-level work and blocking on it."""
        app_config.update(concurrency_per_instance=2)
        pageindex.reconfigure_clients(["http://a:1"])
        work, coord = pageindex.work_pool(), pageindex.coordination_pool()

        def node_call():
            return 1

        def per_document(_):
            return sum(f.result(timeout=10)
                       for f in [work.submit(node_call) for _ in range(4)])

        futures = [coord.submit(per_document, i) for i in range(5)]
        assert [f.result(timeout=20) for f in futures] == [4] * 5

    def test_same_pool_nesting_would_deadlock(self):
        """Documents the hazard: identical work on ONE pool times out. If this
        ever stops timing out the guard above has become unnecessary."""
        import concurrent.futures as cf
        pool = cf.ThreadPoolExecutor(max_workers=2)
        try:
            def outer(_):
                return pool.submit(lambda: 1).result(timeout=2)
            futures = [pool.submit(outer, i) for i in range(5)]
            with pytest.raises(Exception):
                [f.result(timeout=6) for f in futures]
        finally:
            pool.shutdown(wait=False)
