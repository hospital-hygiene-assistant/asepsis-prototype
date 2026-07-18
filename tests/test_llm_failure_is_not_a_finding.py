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
import time
from fastapi.testclient import TestClient


from pageindex import PassageDecisionKind, QuestionRun
from pageindex import search as pi_search
from pageindex.nodes import PageNode
import server
from api.routers import chat as chat_router
from api.question_answering import AnswerKind, AnswerOutcome
from api.retrieval import (
    DocumentSearch,
    LibrarySearchResult,
    LibraryStatus,
)

GENERATION_ID = "a" * 64


@pytest.fixture
def client():
    return TestClient(server.app)


def leaf(node_id="glycaemic-targets"):
    return PageNode(
        node_id=node_id, title="Glycaemic Targets", heading_level=2,
        line_idx=8, summary="s", content="HbA1c target is <53 mmol/mol.",
    )


def evaluate(target=None):
    target = target or leaf()
    run = QuestionRun()
    run.set_total(1)
    return pi_search._evaluate_leaf(
        target, "q", "doc", "crumb", "parent", run=run
    )


class TestUnevaluatedLeafIsNotRejected:
    def test_model_failure_yields_error_not_rejected(self):
        with patch.object(pi_search, "_chat", side_effect=RuntimeError("llama runner died")):
            result = evaluate()
        # "rejected" would assert the passage is clinically irrelevant.
        assert result.kind.audit_status == "error"
        assert result.kind.relevant is False

    def test_failure_reason_is_carried_not_silently_dropped(self):
        with patch.object(pi_search, "_chat", side_effect=RuntimeError("llama runner died")):
            result = evaluate()
        assert "llama runner died" in result.reason

    def test_unparseable_verdict_is_also_an_error(self):
        # The model answered, but with nothing usable. Still not a judgement.
        with patch.object(pi_search, "_chat", return_value="I'm afraid I can't do that"):
            with patch.object(pi_search, "_parse_json_response", return_value=None):
                result = evaluate()
        assert result.kind.audit_status == "error"

    def test_errored_leaves_are_tracked_apart_from_rejected(self):
        run = QuestionRun()
        run.set_total(1)
        with patch.object(pi_search, "_chat", side_effect=RuntimeError("down")):
            pi_search._evaluate_leaf(leaf("n1"), "q", "doc", "crumb", "parent", run=run)
        events = run.events()
        assert "doc::n1" in events["errored"]
        assert "doc::n1" not in events["rejected"]

    def test_a_real_rejection_still_rejects(self):
        # The honest negative must survive: this is a genuine model verdict.
        with patch.object(pi_search, "_chat", return_value='{"relevant": false, "reason": "off topic"}'):
            result = evaluate()
        assert result.kind is PassageDecisionKind.PASSAGE_REJECTED
        assert result.reason == "off topic"

    def test_a_real_hit_still_retrieves(self):
        # The quote is copied verbatim out of leaf().content on purpose: the
        # design says a leaf earns "retrieved" by quoting its own text, so the
        # fixture for a *good* verdict must not be one that fails that rule.
        with patch.object(
            pi_search, "_chat",
            return_value='{"relevant": true, "reason": "states the target", '
                         '"quote": "HbA1c target is <53"}',
        ):
            result = evaluate()
        assert result.kind is PassageDecisionKind.PASSAGE_RETRIEVED
        assert result.quote == "HbA1c target is <53"
        assert result.quote in leaf().content


class TestChatEndpointRefusesToFakeAFinding:
    @staticmethod
    def _answering(outcome):
        return lambda question, run: outcome

    @staticmethod
    def _search(status):
        complete = status is LibraryStatus.COMPLETE
        return LibrarySearchResult(
            query="What PPE for MRSA?",
            generation_id=GENERATION_ID if complete else None,
            documents=(
                DocumentSearch(
                    "hygiene", (), (), ()
                ),
            ) if complete else (),
        )

    def test_total_failure_is_503_not_insufficient_evidence(self, client, monkeypatch):
        """The exact observed bug."""
        outcome = AnswerOutcome.retrieval_unavailable(
            self._search(LibraryStatus.UNAVAILABLE)
        )
        monkeypatch.setattr(
            chat_router, "answer_question", self._answering(outcome)
        )
        response = client.post("/api/chat", json={"query": "What PPE for MRSA?"})

        body = str(self._result(client, response)["result"])
        assert "judged relevant" not in body, "must not state a finding about the documents"
        assert "insufficient_evidence" not in body

    def test_the_error_is_a_machine_readable_technical_outcome(self, client, monkeypatch):
        outcome = AnswerOutcome.retrieval_unavailable(
            self._search(LibraryStatus.UNAVAILABLE)
        )
        monkeypatch.setattr(
            chat_router, "answer_question", self._answering(outcome)
        )
        response = client.post("/api/chat", json={"query": "q"})
        body = self._result(client, response)["result"]
        assert body["outcome"]["kind"] == "retrieval_unavailable"
        assert body["outcome"]["coverage"] == {
            "status": "unavailable",
            "generation_id": None,
            "searched_documents": 0,
            "total_documents": 0,
            "incomplete_checks": 0,
        }

    def test_genuine_empty_result_still_reports_insufficient_evidence(self, client, monkeypatch):
        """A real "nothing matched" must survive — every leaf was checked."""
        outcome = AnswerOutcome.insufficient_evidence(
            self._search(LibraryStatus.COMPLETE)
        )
        monkeypatch.setattr(
            chat_router, "answer_question", self._answering(outcome)
        )
        response = client.post("/api/chat", json={"query": "unrelated question"})
        body = self._result(client, response)["result"]
        assert body["outcome"]["kind"] == "insufficient_evidence"

    def test_a_wire_defect_after_search_is_not_relabeled_as_retrieval_unavailable(
        self, monkeypatch
    ):
        outcome = AnswerOutcome.insufficient_evidence(
            self._search(LibraryStatus.COMPLETE)
        )
        monkeypatch.setattr(
            chat_router, "answer_question", self._answering(outcome)
        )
        real_encode = chat_router.encode_outcome

        def fail_only_for_completed_search(result, *, run_id):
            if result.kind is AnswerKind.INSUFFICIENT_EVIDENCE:
                raise RuntimeError("producer defect")
            return real_encode(result, run_id=run_id)

        monkeypatch.setattr(chat_router, "encode_outcome", fail_only_for_completed_search)
        response = TestClient(
            server.app, raise_server_exceptions=False
        ).post("/api/chat", json={"query": "q"})

        body = self._result(TestClient(server.app), response)
        assert body["phase"] == "failed"
        assert body["error"] == {"code": "unexpected_failure"}
        assert "retrieval_unavailable" not in str(body)
    @staticmethod
    def _result(client, response):
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + 1
        while True:
            body = client.get(f"/api/runs/{run_id}").json()
            if body["phase"] in {"completed", "failed", "cancelled", "timed_out"}:
                return body
            assert time.monotonic() < deadline
            time.sleep(0.005)
