"""Two clients retrieving at once must not see each other's run.

Progress and per-node verdicts used to be process-wide: the second client's
start reset the first's counters, and a single unscoped /api/status handed
whoever asked a blend of both — including reasons and verbatim quotes derived
from the other's question.
"""

import threading
import time

import pytest
from fastapi.testclient import TestClient


import server
from api.runs import RunRegistry
from pageindex import PassageDecision, PassageDecisionKind, QuestionRun


def retrieved(
    node_id: str, reason: str = "", quote: str = "", document_id: str = ""
):
    return PassageDecision(
        node_id,
        PassageDecisionKind.PASSAGE_RETRIEVED,
        reason,
        quote,
        document_id=document_id,
    )


@pytest.fixture
def client():
    return TestClient(server.app)


class TestQuestionRunIsolation:
    def test_a_passage_decision_updates_audit_and_progress_atomically(self):
        run = QuestionRun()
        run.set_total(1)
        run.record_decision(retrieved("p", "exact", "verbatim"))

        assert run.progress() == {"total": 1, "done": 1}
        assert run.events()["meta"]["p"] == {
            "status": "retrieved",
            "reason": "exact",
            "quote": "verbatim",
        }

    def test_a_failed_section_check_is_kept_but_audited_as_error(self):
        run = QuestionRun()
        run.record_decision(PassageDecision(
            "section",
            PassageDecisionKind.SECTION_CHECK_FAILED,
            "model unavailable",
        ))

        assert run.events()["kept"] == ["section"]
        assert run.events()["meta"]["section"]["status"] == "error"

    def test_a_passage_cannot_complete_twice(self):
        run = QuestionRun()
        run.set_total(1)
        run.record_decision(retrieved("p", "first", "quote"))

        with pytest.raises(RuntimeError, match="twice"):
            run.record_decision(retrieved("p", "second", "replacement"))

        assert run.progress() == {"total": 1, "done": 1}
        assert run.events()["meta"]["p"]["reason"] == "first"

    def test_the_same_node_identity_can_complete_in_two_documents(self):
        run = QuestionRun()
        run.set_total(2)
        run.record_decision(retrieved("ppe", document_id="guide-a"))
        run.record_decision(retrieved("ppe", document_id="guide-b"))

        assert run.progress() == {"total": 2, "done": 2}
        assert [
            (item["document_id"], item["node_id"])
            for item in run.events()["decisions"]
        ] == [("guide-a", "ppe"), ("guide-b", "ppe")]
        assert set(run.events()["meta"]) == {"guide-a::ppe", "guide-b::ppe"}

    def test_two_runs_keep_their_own_progress(self):
        a, b = QuestionRun(), QuestionRun()
        a.set_total(10)
        b.set_total(3)
        a.record_decision(retrieved("a"))
        assert a.progress() == {"total": 10, "done": 1}
        assert b.progress() == {"total": 3, "done": 0}

    def test_starting_one_run_does_not_reset_another(self):
        a = QuestionRun()
        a.set_total(5)
        a.record_decision(retrieved("a"))
        QuestionRun().set_total(99)   # a second client arrives mid-run
        assert a.progress() == {"total": 5, "done": 1}

    def test_verdicts_do_not_leak_between_runs(self):
        a, b = QuestionRun(), QuestionRun()
        a.set_total(1)
        b.set_total(1)
        a.record_decision(retrieved("a-node", "because of A's question", "A's quote"))
        b.record_decision(PassageDecision(
            "b-node", PassageDecisionKind.PASSAGE_REJECTED
        ))

        assert a.events()["retrieved"] == ["a-node"]
        assert b.events()["retrieved"] == []
        # The quote is derived from the asker's question, so it must not travel.
        assert "a-node" not in b.events()["meta"]

    def test_concurrent_writers_do_not_lose_records(self):
        run = QuestionRun()
        run.set_total(200)

        def worker(offset: int):
            for i in range(100):
                run.record_decision(retrieved(f"n{offset + i}"))

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
        run.begin_retrieval("Reading…")
        run.set_total(7)
        run.record_decision(retrieved("passage"))

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
