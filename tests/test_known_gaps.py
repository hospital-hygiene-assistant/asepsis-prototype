"""The CRITICAL findings, as executable documentation.

Every test here asserts an invariant the system *claims* to hold and does not.
They are `xfail(strict=True)`, so each one fails today and the suite stays green
— but the day someone makes one pass, it turns into a hard failure that says
"this is fixed, come and update me". That is the point: a finding in a markdown
file rots, a finding in the suite cannot.

The invariants are deliberately decision-neutral. None of them presumes *how*
the gap gets closed, only that the claim becomes true. See
ARCHITECTURE_REVIEW_2026-07-14.md for the full findings and the open questions
that belong to Federico.
"""

import json

import pytest
from unittest.mock import patch

from api import pdf
from api.pdf import _normalized_bbox
from pageindex import RunState
from pageindex import search as pi_search
from api.runs import RunRegistry
from pageindex.nodes import PageNode, _node_to_dict

LEAF = PageNode(
    node_id="glycaemic-targets", title="Glycaemic Targets", heading_level=2,
    line_idx=8, summary="s", content="HbA1c target is <53 mmol/mol.",
)


class TestC1QuoteIsNeverVerified:
    """`prompts.py` asks for a character-for-character copy, `search.py`'s
    docstring says a leaf earns "retrieved" by quoting, and /api/chat tells the
    clinician "every source below was selected with a verbatim quote". Nothing
    checks it.
    """

    @pytest.mark.xfail(strict=True, reason="C1: a fabricated quote is accepted as a cited source")
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

    @pytest.mark.xfail(strict=True, reason="C1: an empty quote is accepted as a cited source")
    def test_a_leaf_with_no_quote_is_not_called_retrieved(self):
        with patch.object(pi_search, "_chat",
                          return_value='{"relevant": true, "reason": "trust me"}'):
            _, result = pi_search._evaluate_leaf(LEAF, "q", "doc", "crumb", "parent", run=RunState())
        assert result["status"] != "retrieved" or result["quote"]


class TestC2HighlightFramesTheHeading:
    """A pin carries two geometries: `bbox` is the heading block,
    `regions` is the per-page union of the body text underneath — where the
    quote actually is. `_normalized_bbox` reads only `bbox`.

    Both halves are already tested and both pass: test_betteringest_ingest
    proves `regions` is produced correctly, TestNormalizedBbox proves `bbox` is
    normalised correctly. The bug lives in the seam between them, which nothing
    covers — every existing bbox test passes a pin with no `regions` at all.
    """

    @pytest.fixture(autouse=True)
    def square_page(self, monkeypatch):
        monkeypatch.setattr(pdf, "_rendered_page_size", lambda stem, page: (1000.0, 1000.0))

    @pytest.mark.xfail(strict=True, reason="C2: the highlight is derived from the heading bbox")
    def test_the_highlight_frames_the_body_not_the_heading(self):
        pin = {
            "page": 1,
            "bbox": [50, 80, 400, 110],            # the heading, near the top
            "regions": [[1, 50, 700, 500, 900]],   # the quoted body, far below it
        }
        box = _normalized_bbox("doc", pin)
        # The practitioner clicks to check the citation; the box must land on the
        # passage that was quoted, not on the section title above it.
        assert box["y"] == pytest.approx(0.7)

    @pytest.mark.xfail(strict=True, reason="C2: only pin['page'] (the heading's page) is ever rendered")
    def test_a_body_that_runs_onto_the_next_page_is_highlighted_there(self):
        pin = {
            "page": 1,
            "bbox": [50, 900, 400, 950],           # heading at the foot of page 1
            "regions": [[2, 50, 100, 500, 400]],   # body continues on page 2
        }
        assert _normalized_bbox("doc", pin)["page"] == 2


class TestSectionPrunedWithoutAVerdict:
    """Narrow, and found while covering the engine — not one of the reviewed
    CRITICALs, recorded here because it is the same family.

    The two common failures are handled correctly: a connection error and an
    unparseable reply both fail open and keep the section. But a reply that *is*
    valid JSON and is not an object — an array, a bare string — falls past the
    `isinstance(result, dict)` guard to verdict=False and drops the entire branch
    with reason='', which reads downstream exactly like a genuine "off topic".
    """

    @pytest.mark.xfail(strict=True, reason="a non-object JSON reply prunes the branch as if judged")
    def test_a_reply_that_is_not_a_verdict_does_not_prune(self):
        node = PageNode(node_id="isolation", title="Isolation", heading_level=1, line_idx=0,
                        summary="s", children=[LEAF])
        with patch.object(pi_search, "_chat", return_value='["not", "a", "verdict"]'):
            verdict, _ = pi_search._check_section_relevant(node, "q", "crumb", run=RunState())
        assert verdict is True, "nothing was judged, so nothing may be pruned"


class TestM2ADroppedDocumentLeavesNoTrace:
    """`run_retrieval` skips a document whose index has gone missing with a bare
    `continue`. It never enters the results, so it is not counted by
    count_eval_errors, not named in the summary, and not visible anywhere: the
    question was answered from a corpus that quietly lost a document, and the
    response looks identical to one where the whole library was read.
    """

    @pytest.mark.xfail(strict=True, reason="M2: a skipped document is not surfaced anywhere")
    def test_a_document_that_could_not_be_read_is_reported(self, tmp_path, monkeypatch):
        from api import retrieval as api_retrieval

        monkeypatch.setattr(api_retrieval, "INDEX_DIR", tmp_path)
        root = PageNode(node_id="root", title="Guideline", heading_level=1, line_idx=0,
                        summary="s", children=[LEAF])
        for stem in ("present", "vanished"):
            (tmp_path / f"{stem}.json").write_text(
                json.dumps([_node_to_dict(root)]), encoding="utf-8")

        class OneDocVanishes:
            def retrieve_with_metadata(self, doc_name, query, state):
                if doc_name == "vanished":
                    raise FileNotFoundError("index deleted mid-run")
                return [], {}

        run = RunRegistry().create("probe")
        results = api_retrieval.run_retrieval("q", OneDocVanishes(), run)
        # Neutral on the fix: name it in the results, count it as an error, or
        # fold it into the partial-grounding caveat — but it must not vanish.
        assert "vanished" in json.dumps(results), "the lost document leaves no trace"


class TestC3RetrieveCannotSayItDidNotSearch:
    """`retrieve()` returns `list[PageNode]`, and an empty list means two
    irreconcilable things: the library was searched and holds nothing relevant,
    or it was never searched at all.

    /api/chat reconstructs the difference by counting error statuses, so the HTTP
    route is safe. Nothing else is: with Ollama stopped, `pipeline.py query`
    prints "No relevant passages found." — a statement about the documents.
    """

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

    @pytest.mark.xfail(strict=True, reason="C3: the engine returns [] instead of refusing to answer")
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
