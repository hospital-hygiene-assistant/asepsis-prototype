"""An unreachable model must never be reported as a clinical finding.

Observed for real: Ollama died mid-run, every leaf eval raised, each was
recorded as status "rejected", the source list came back empty, and /api/chat
answered 200 with:

    status:  insufficient_evidence
    summary: "No passage in the library was judged relevant to this question."

That is a statement about the documents. Nothing had been read. A practitioner
could reasonably conclude no guideline covers their question.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tauri-app"))

import pageindex  # noqa: E402
import server  # noqa: E402
from server import _count_eval_errors  # noqa: E402


@pytest.fixture
def client():
    return TestClient(server.app)


def leaf(node_id="glycaemic-targets"):
    return pageindex.PageNode(
        node_id=node_id, title="Glycaemic Targets", heading_level=2,
        line_idx=8, summary="s", content="HbA1c target is <53 mmol/mol.",
    )


class TestUnevaluatedLeafIsNotRejected:
    def test_model_failure_yields_error_not_rejected(self):
        with patch.object(pageindex, "_chat", side_effect=RuntimeError("llama runner died")):
            _, result = pageindex._evaluate_leaf(leaf(), "q", "doc", "crumb", "parent")
        # "rejected" would assert the passage is clinically irrelevant.
        assert result["status"] == "error"
        assert result["relevant"] is False

    def test_failure_reason_is_carried_not_silently_dropped(self):
        with patch.object(pageindex, "_chat", side_effect=RuntimeError("llama runner died")):
            _, result = pageindex._evaluate_leaf(leaf(), "q", "doc", "crumb", "parent")
        assert "llama runner died" in result["reason"]

    def test_unparseable_verdict_is_also_an_error(self):
        # The model answered, but with nothing usable. Still not a judgement.
        with patch.object(pageindex, "_chat", return_value="I'm afraid I can't do that"):
            with patch.object(pageindex, "_parse_json_response", return_value=None):
                _, result = pageindex._evaluate_leaf(leaf(), "q", "doc", "crumb", "parent")
        assert result["status"] == "error"

    def test_errored_leaves_are_tracked_apart_from_rejected(self):
        pageindex.start_run(1)
        with patch.object(pageindex, "_chat", side_effect=RuntimeError("down")):
            pageindex._evaluate_leaf(leaf("n1"), "q", "doc", "crumb", "parent")
        events = pageindex.get_live_events()
        assert "n1" in events["errored"]
        assert "n1" not in events["rejected"]

    def test_a_real_rejection_still_rejects(self):
        # The honest negative must survive: this is a genuine model verdict.
        with patch.object(pageindex, "_chat", return_value='{"relevant": false, "reason": "off topic"}'):
            _, result = pageindex._evaluate_leaf(leaf(), "q", "doc", "crumb", "parent")
        assert result["status"] == "rejected"
        assert result["reason"] == "off topic"

    def test_a_real_hit_still_retrieves(self):
        with patch.object(
            pageindex, "_chat",
            return_value='{"relevant": true, "reason": "states the target", "quote": "HbA1c <53"}',
        ):
            _, result = pageindex._evaluate_leaf(leaf(), "q", "doc", "crumb", "parent")
        assert result["status"] == "retrieved"
        assert result["quote"] == "HbA1c <53"


class TestCountEvalErrors:
    def test_counts_across_documents(self):
        results = {
            "doc_a": {"node_meta": {"n1": {"status": "error"}, "n2": {"status": "rejected"}}},
            "doc_b": {"node_meta": {"n3": {"status": "error"}, "n4": {"status": "retrieved"}}},
        }
        assert _count_eval_errors(results) == 2

    def test_zero_when_everything_was_evaluated(self):
        results = {"doc_a": {"node_meta": {"n1": {"status": "retrieved"}, "n2": {"status": "rejected"}}}}
        assert _count_eval_errors(results) == 0

    @pytest.mark.parametrize("results", [{}, {"doc": {}}, {"doc": {"node_meta": None}}])
    def test_tolerates_missing_metadata(self, results):
        assert _count_eval_errors(results) == 0


class TestChatEndpointRefusesToFakeAFinding:
    def test_total_failure_is_503_not_insufficient_evidence(self, client, monkeypatch):
        """The exact observed bug."""
        monkeypatch.setattr(server, "_run_retrieval", lambda *a, **k: {
            "doc": {"nodes": [], "tree": [], "retrieved_ids": [], "node_meta": {"n1": {"status": "error"}, "n2": {"status": "error"}}},
        })
        response = client.post("/api/chat", json={"query": "What PPE for MRSA?"})

        assert response.status_code == 503
        body = response.text
        assert "judged relevant" not in body, "must not state a finding about the documents"
        assert "insufficient_evidence" not in body

    def test_the_error_says_it_is_not_a_finding(self, client, monkeypatch):
        monkeypatch.setattr(server, "_run_retrieval", lambda *a, **k: {
            "doc": {"nodes": [], "tree": [], "retrieved_ids": [], "node_meta": {"n1": {"status": "error"}}},
        })
        response = client.post("/api/chat", json={"query": "q"})
        assert "not a finding" in response.json()["error"]

    def test_genuine_empty_result_still_reports_insufficient_evidence(self, client, monkeypatch):
        """A real "nothing matched" must survive — every leaf was checked."""
        monkeypatch.setattr(server, "_run_retrieval", lambda *a, **k: {
            "doc": {"nodes": [], "tree": [], "retrieved_ids": [], "node_meta": {"n1": {"status": "rejected"}, "n2": {"status": "rejected"}}},
        })
        response = client.post("/api/chat", json={"query": "unrelated question"})
        assert response.status_code == 200
        assert response.json()["grounding"]["status"] == "insufficient_evidence"
