"""Two clients retrieving at once must not see each other's run.

Progress and per-node verdicts used to be process-wide: the second client's
start reset the first's counters, and a single unscoped /api/status handed
whoever asked a blend of both — including reasons and verbatim quotes derived
from the other's question.
"""

import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tauri-app"))

import server  # noqa: E402
from api.runs import Run, RunRegistry  # noqa: E402
from pageindex import RunState  # noqa: E402


@pytest.fixture
def client():
    return TestClient(server.app)


class TestRunStateIsolation:
    def test_two_runs_keep_their_own_progress(self):
        a, b = RunState(), RunState()
        a.start(10)
        b.start(3)
        a.leaf_done()
        assert a.progress() == {"total": 10, "done": 1}
        assert b.progress() == {"total": 3, "done": 0}

    def test_starting_one_run_does_not_reset_another(self):
        a = RunState()
        a.start(5)
        a.leaf_done()
        RunState().start(99)          # a second client arrives mid-run
        assert a.progress() == {"total": 5, "done": 1}

    def test_verdicts_do_not_leak_between_runs(self):
        a, b = RunState(), RunState()
        a.mark("retrieved", "a-node")
        a.set_meta("a-node", "retrieved", "because of A's question", quote="A's quote")
        b.mark("rejected", "b-node")

        assert a.events()["retrieved"] == ["a-node"]
        assert b.events()["retrieved"] == []
        # The quote is derived from the asker's question, so it must not travel.
        assert "a-node" not in b.events()["meta"]

    def test_concurrent_writers_do_not_lose_records(self):
        run = RunState()
        run.start(200)

        def worker(offset: int):
            for i in range(100):
                run.mark("retrieved", f"n{offset + i}")
                run.leaf_done()

        threads = [threading.Thread(target=worker, args=(o,)) for o in (0, 1000)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert run.progress()["done"] == 200
        assert len(run.events()["retrieved"]) == 200


class TestRegistry:
    def test_a_run_is_found_by_its_id(self):
        registry = RunRegistry()
        run = registry.create("mine")
        assert registry.get("mine") is run

    def test_unknown_ids_are_not_invented(self):
        assert RunRegistry().get("nope") is None

    def test_ids_are_generated_when_the_client_offers_none(self):
        registry = RunRegistry()
        first, second = registry.create(), registry.create()
        assert first.id and second.id and first.id != second.id

    def test_expired_runs_are_dropped(self):
        registry = RunRegistry(ttl=0.05)
        registry.create("old")
        time.sleep(0.06)
        registry.create("new")
        assert registry.get("old") is None
        assert registry.get("new") is not None

    def test_the_registry_is_bounded(self):
        registry = RunRegistry(max_runs=4)
        for i in range(20):
            registry.create(f"run-{i}")
        assert len(registry._runs) <= 4

    def test_the_newest_run_survives_eviction(self):
        registry = RunRegistry(max_runs=3)
        for i in range(10):
            registry.create(f"run-{i}")
        assert registry.get("run-9") is not None


class TestRunsEndpoint:
    def test_a_run_reports_its_own_progress(self, client):
        from api.runs import registry
        run = registry.create("probe")
        run.set_phase("retrieval", "Reading…")
        run.state.start(7)
        run.state.leaf_done()

        body = client.get("/api/runs/probe").json()
        assert body["run_id"] == "probe"
        assert body["phase"] == "retrieval"
        assert body["progress"] == {"total": 7, "done": 1}

    def test_an_unknown_run_is_404(self, client):
        assert client.get("/api/runs/never-started").status_code == 404

    def test_status_no_longer_carries_run_data(self, client):
        # It reports the service. A run belongs to whoever started it.
        body = client.get("/api/status").json()
        assert "instances" in body and "any_busy" in body
        assert "progress" not in body
        assert "live" not in body
        assert "chat" not in body
