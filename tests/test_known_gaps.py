"""Reviewed architecture findings as executable documentation.

Resolved invariants stay here as regressions. A genuinely open finding uses
`xfail(strict=True)`, so an implementation silently landing later becomes a
hard failure that requires this inventory to be updated.

The invariants are deliberately decision-neutral. None of them presumes *how*
the gap gets closed, only that the claim becomes true. See
ARCHITECTURE_REVIEW_2026-07-14.md for the full findings and the open questions
that belong to Federico.
"""

import json

import pytest
from unittest.mock import patch

from pageindex import RunState
from pageindex import search as pi_search
from api.runs import RunRegistry
from pageindex.nodes import PageNode, _node_to_dict

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
            _, result = pi_search._evaluate_leaf(LEAF, "q", "doc", "crumb", "parent", run=RunState())
        # Neutral on the fix: drop the leaf, or mark it unverified and stop
        # claiming it — either way, nothing still called "retrieved" may carry a
        # quote that is not in the passage it points at.
        if result["status"] == "retrieved":
            assert result["quote"] in LEAF.content

    def test_a_leaf_with_no_quote_is_not_called_retrieved(self):
        with patch.object(pi_search, "_chat",
                          return_value='{"relevant": true, "reason": "trust me"}'):
            _, result = pi_search._evaluate_leaf(LEAF, "q", "doc", "crumb", "parent", run=RunState())
        assert result["status"] != "retrieved" or result["quote"]


class TestSectionWithoutAVerdictFailsOpen:
    """Malformed model output cannot prune an unjudged document branch."""

    def test_a_reply_that_is_not_a_verdict_does_not_prune(self):
        node = PageNode(node_id="isolation", title="Isolation", heading_level=1, line_idx=0,
                        summary="s", children=[LEAF])
        with patch.object(pi_search, "_chat", return_value='["not", "a", "verdict"]'):
            verdict, _ = pi_search._check_section_relevant(node, "q", "crumb", run=RunState())
        assert verdict is True, "nothing was judged, so nothing may be pruned"


class TestM2UnavailableDocumentRemainsVisible:
    """A document that cannot be searched remains named in coverage truth."""

    def test_a_document_that_could_not_be_read_is_reported(self, tmp_path, monkeypatch):
        from api import retrieval as api_retrieval

        monkeypatch.setattr(api_retrieval, "INDEX_DIR", tmp_path)
        root = PageNode(node_id="root", title="Guideline", heading_level=1, line_idx=0,
                        summary="s", children=[LEAF])
        for stem in ("present", "vanished"):
            (tmp_path / f"{stem}.json").write_text(
                json.dumps([_node_to_dict(root)]), encoding="utf-8")

        class OneDocVanishes:
            def retrieve_with_metadata_from_path(
                self, doc_name, query, state, index_path, model=None
            ):
                if doc_name == "vanished":
                    raise FileNotFoundError("index deleted mid-run")
                return [], {}

        run = RunRegistry().create("probe")
        results = api_retrieval.WholeLibraryRetrieval(
            OneDocVanishes(), index_dir=tmp_path
        ).search("q", run)
        # Neutral on the fix: name it in the results, count it as an error, or
        # fold it into the partial-grounding caveat — but it must not vanish.
        assert results.document("vanished") is not None
        assert any(item.document == "vanished" for item in results.diagnostics)


class TestH6InstanceCountIsUnbounded:
    """`ConfigRequest.ollama_instances` is a bare `int` with no Field bound
    (tauri-app/api/routers/status.py), and set_ollama_instances only clamps the
    lower end. One unauthenticated POST asking for 500 will try to fork ~499
    `ollama serve` processes, each an independent runtime with its own memory.

    Low risk on a laptop behind localhost; an unauthenticated resource-exhaustion
    primitive the day this has a URL. The fix is one line: Field(ge=1, le=8).
    """

    @pytest.mark.xfail(strict=True, reason="H6: ollama_instances has no upper bound")
    def test_an_absurd_instance_count_is_refused(self, monkeypatch):
        import server
        from api.routers import status as status_router
        from fastapi.testclient import TestClient

        # Never actually spawn: the point is that the request is not rejected
        # before it ever reaches the spawner.
        monkeypatch.setattr(status_router, "set_ollama_instances",
                            lambda n: {"requested": n, "live": 1, "urls": [], "errors": []})
        response = TestClient(server.app).post("/api/config", json={"ollama_instances": 500})
        assert response.status_code == 422, "a request to fork 499 processes should not validate"


class TestC3RetrieveCannotSayItDidNotSearch:
    """The lossy low-level interface refuses to flatten an incomplete search."""

    @pytest.fixture
    def index_of_two_leaves(self, tmp_path, monkeypatch):
        """Written through the real serialiser, not hand-rolled JSON.

        A hand-written tree silently used snake_case keys and every read raised
        KeyError — which `pytest.raises(Exception)` then caught and reported as
        the bug being reproduced. Round-tripping real PageNodes cannot drift.
        """
        monkeypatch.setattr(pi_search, "INDEX_DIR", tmp_path)
        root = PageNode(
            node_id="root", title="Guideline", heading_level=1, line_idx=0, summary="root",
            children=[
                PageNode(node_id="hand-hygiene", title="Hand Hygiene", heading_level=2,
                         line_idx=4, summary="s", content="Disinfect hands."),
                PageNode(node_id="isolation", title="Isolation", heading_level=2,
                         line_idx=9, summary="s", content="Single room."),
            ],
        )
        (tmp_path / "hygiene.json").write_text(
            json.dumps([_node_to_dict(root)]), encoding="utf-8")
        return tmp_path

    def test_an_unreachable_model_is_not_an_empty_result(self, index_of_two_leaves):
        down = RuntimeError("Failed to connect to Ollama.")
        with patch.object(pi_search, "_chat", side_effect=down):
            try:
                nodes = pi_search.retrieve("hygiene", "MRSA?", RunState())
            except Exception:
                return  # Refusing loudly is a valid fix; its shape is Federico's call.
        # Neutral on the fix: raise, or return something that carries "not
        # searched". What must not happen is an empty list, which every caller
        # reads as "the library holds no guidance on this".
        assert nodes != [], "an empty list is indistinguishable from a genuine absence"

    def test_a_genuine_absence_still_returns_empty(self, index_of_two_leaves):
        """The honest negative must survive whatever fix lands above.

        Doubles as the fixture's own guard: if the tree stopped deserialising,
        this fails loudly instead of quietly propping up the xfail above.
        """
        with patch.object(pi_search, "_chat",
                          return_value='{"relevant": false, "reason": "off topic"}'):
            assert pi_search.retrieve("hygiene", "unrelated", RunState()) == []
