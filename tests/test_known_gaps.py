"""Reviewed architecture findings as executable documentation.

Resolved invariants stay here as regressions. A genuinely open finding uses
`xfail(strict=True)`, so an implementation silently landing later becomes a
hard failure that requires this inventory to be updated.

The invariants are deliberately decision-neutral. None of them presumes *how*
the gap gets closed, only that the claim becomes true. See
ARCHITECTURE_REVIEW_2026-07-14.md for the full findings and the open questions
that belong to Federico.
"""

from unittest.mock import patch

from pageindex import QuestionRun
from pageindex import search as pi_search
from api.runs import RunRegistry
from pageindex.library import ExpectedLibraryStore, LibraryCandidate
from pageindex.nodes import PageNode

LEAF = PageNode(
    node_id="glycaemic-targets", title="Glycaemic Targets", heading_level=2,
    line_idx=8, summary="s", content="HbA1c target is <53 mmol/mol.",
)


class TestC1QuoteVerification:
    """Only a non-empty verbatim quote can become verified evidence."""

    def test_a_retrieved_leaf_quotes_its_own_text(self):
        fabricated = ('{"relevant": true, "reason": "states the rule", '
                      '"quote": "Isolate in a negative-pressure room for 14 days"}')
        with patch.object(pi_search, "_chat", return_value=fabricated):
            _, result = pi_search._evaluate_leaf(LEAF, "q", "doc", "crumb", "parent", run=QuestionRun())
        # Neutral on the fix: drop the leaf, or mark it unverified and stop
        # claiming it — either way, nothing still called "retrieved" may carry a
        # quote that is not in the passage it points at.
        if result["status"] == "retrieved":
            assert result["quote"] in LEAF.content

    def test_a_leaf_with_no_quote_is_not_called_retrieved(self):
        with patch.object(pi_search, "_chat",
                          return_value='{"relevant": true, "reason": "trust me"}'):
            _, result = pi_search._evaluate_leaf(LEAF, "q", "doc", "crumb", "parent", run=QuestionRun())
        assert result["status"] != "retrieved" or result["quote"]


class TestSectionWithoutAVerdictFailsOpen:
    """Malformed model output cannot prune an unjudged document branch."""

    def test_a_reply_that_is_not_a_verdict_does_not_prune(self):
        node = PageNode(node_id="isolation", title="Isolation", heading_level=1, line_idx=0,
                        summary="s", children=[LEAF])
        with patch.object(pi_search, "_chat", return_value='["not", "a", "verdict"]'):
            verdict, _ = pi_search._check_section_relevant(node, "q", "crumb", run=QuestionRun())
        assert verdict is True, "nothing was judged, so nothing may be pruned"


class TestM2UnavailableDocumentRemainsVisible:
    """A document that cannot be searched remains named in coverage truth."""

    def test_a_document_that_could_not_be_read_is_reported(self, tmp_path):
        from api import retrieval as api_retrieval

        library = ExpectedLibraryStore(tmp_path / "library")
        library.publish({
            "present": LibraryCandidate("# Present\n\nGuidance."),
            "vanished": LibraryCandidate("# Vanished\n\nGuidance."),
        })

        class OneDocVanishes:
            def retrieve_with_metadata(
                self, doc_name, query, run, index, **_options
            ):
                if doc_name == "vanished":
                    raise FileNotFoundError("index deleted mid-run")
                return [], {}

        run = RunRegistry().create("probe")
        results = api_retrieval.WholeLibraryRetrieval(
            OneDocVanishes(), library_store=library
        ).search("q", run)
        # Neutral on the fix: name it in the results, count it as an error, or
        # fold it into the partial-grounding caveat — but it must not vanish.
        assert results.document("vanished") is not None
        assert any(item.document == "vanished" for item in results.diagnostics)


class TestInstanceCountValidation:
    """The HTTP interface refuses unsafe Ollama pool sizes before spawning."""

    def test_an_absurd_instance_count_is_refused(self, monkeypatch):
        import server
        from api.routers import status as status_router
        from fastapi.testclient import TestClient

        # Never actually spawn: the point is that the request is not rejected
        # before it ever reaches the spawner.
        monkeypatch.setattr(status_router, "set_ollama_instances",
                            lambda n: {"requested": n, "live": 1, "urls": [], "errors": []})
        response = TestClient(server.app).post(
            "/api/config", json={"ollama_instances": 500}
        )
        assert response.status_code == 422
