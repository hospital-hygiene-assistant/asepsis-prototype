"""An unreachable model must never be reported as a clinical finding.

Observed for real: Ollama died mid-run, every leaf eval raised, each was
recorded as status "rejected", the source list came back empty, and /api/chat
answered 200 with:

    status:  insufficient_evidence
    summary: "No passage in the library was judged relevant to this question."

That is a statement about the documents. Nothing had been read. A practitioner
could reasonably conclude no guideline covers their question.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


from pageindex import RunState
from pageindex import search as pi_search
from pageindex.nodes import PageNode
import server
from api.routers import chat as chat_router
from api.question_answering import AnswerKind, AnswerOutcome
from api.retrieval import LibrarySearchResult, LibraryStatus


@pytest.fixture
def client():
    return TestClient(server.app)


def leaf(node_id="glycaemic-targets"):
    return PageNode(
        node_id=node_id, title="Glycaemic Targets", heading_level=2,
        line_idx=8, summary="s", content="HbA1c target is <53 mmol/mol.",
    )


class TestUnevaluatedLeafIsNotRejected:
    def test_model_failure_yields_error_not_rejected(self):
        with patch.object(pi_search, "_chat", side_effect=RuntimeError("llama runner died")):
            _, result = pi_search._evaluate_leaf(leaf(), "q", "doc", "crumb", "parent", run=RunState())
        # "rejected" would assert the passage is clinically irrelevant.
        assert result["status"] == "error"
        assert result["relevant"] is False

    def test_failure_reason_is_carried_not_silently_dropped(self):
        with patch.object(pi_search, "_chat", side_effect=RuntimeError("llama runner died")):
            _, result = pi_search._evaluate_leaf(leaf(), "q", "doc", "crumb", "parent", run=RunState())
        assert "llama runner died" in result["reason"]

    def test_unparseable_verdict_is_also_an_error(self):
        # The model answered, but with nothing usable. Still not a judgement.
        with patch.object(pi_search, "_chat", return_value="I'm afraid I can't do that"):
            with patch.object(pi_search, "_parse_json_response", return_value=None):
                _, result = pi_search._evaluate_leaf(leaf(), "q", "doc", "crumb", "parent", run=RunState())
        assert result["status"] == "error"

    def test_errored_leaves_are_tracked_apart_from_rejected(self):
        run = RunState()
        run.start(1)
        with patch.object(pi_search, "_chat", side_effect=RuntimeError("down")):
            pi_search._evaluate_leaf(leaf("n1"), "q", "doc", "crumb", "parent", run=run)
        events = run.events()
        assert "n1" in events["errored"]
        assert "n1" not in events["rejected"]

    def test_a_real_rejection_still_rejects(self):
        # The honest negative must survive: this is a genuine model verdict.
        with patch.object(pi_search, "_chat", return_value='{"relevant": false, "reason": "off topic"}'):
            _, result = pi_search._evaluate_leaf(leaf(), "q", "doc", "crumb", "parent", run=RunState())
        assert result["status"] == "rejected"
        assert result["reason"] == "off topic"

    def test_a_real_hit_still_retrieves(self):
        # The quote is copied verbatim out of leaf().content on purpose: the
        # design says a leaf earns "retrieved" by quoting its own text, so the
        # fixture for a *good* verdict must not be one that fails that rule.
        with patch.object(
            pi_search, "_chat",
            return_value='{"relevant": true, "reason": "states the target", '
                         '"quote": "HbA1c target is <53"}',
        ):
            _, result = pi_search._evaluate_leaf(leaf(), "q", "doc", "crumb", "parent", run=RunState())
        assert result["status"] == "retrieved"
        assert result["quote"] == "HbA1c target is <53"
        assert result["quote"] in leaf().content


class TestChatEndpointRefusesToFakeAFinding:
    @staticmethod
    def _answering(outcome):
        class StubAnswering:
            def answer(self, question, run):
                return outcome

        return lambda run: StubAnswering()

    @staticmethod
    def _search(status):
        return LibrarySearchResult(
            query="What PPE for MRSA?",
            generation_id="generation-test",
            status=status,
            documents=(),
            evidence=(),
            diagnostics=(),
        )

    def test_total_failure_is_503_not_insufficient_evidence(self, client, monkeypatch):
        """The exact observed bug."""
        outcome = AnswerOutcome(
            AnswerKind.RETRIEVAL_UNAVAILABLE,
            self._search(LibraryStatus.UNAVAILABLE),
        )
        monkeypatch.setattr(
            chat_router, "_question_answering", self._answering(outcome)
        )
        response = client.post("/api/chat", json={"query": "What PPE for MRSA?"})

        assert response.status_code == 503
        body = response.text
        assert "judged relevant" not in body, "must not state a finding about the documents"
        assert "insufficient_evidence" not in body

    def test_the_error_is_a_machine_readable_technical_outcome(self, client, monkeypatch):
        outcome = AnswerOutcome(
            AnswerKind.RETRIEVAL_UNAVAILABLE,
            self._search(LibraryStatus.UNAVAILABLE),
        )
        monkeypatch.setattr(
            chat_router, "_question_answering", self._answering(outcome)
        )
        response = client.post("/api/chat", json={"query": "q"})
        assert response.json()["error"]["kind"] == "retrieval_unavailable"

    def test_genuine_empty_result_still_reports_insufficient_evidence(self, client, monkeypatch):
        """A real "nothing matched" must survive — every leaf was checked."""
        outcome = AnswerOutcome(
            AnswerKind.INSUFFICIENT_EVIDENCE,
            self._search(LibraryStatus.COMPLETE),
        )
        monkeypatch.setattr(
            chat_router, "_question_answering", self._answering(outcome)
        )
        response = client.post("/api/chat", json={"query": "unrelated question"})
        assert response.status_code == 200
        assert response.json()["grounding"]["status"] == "insufficient_evidence"
