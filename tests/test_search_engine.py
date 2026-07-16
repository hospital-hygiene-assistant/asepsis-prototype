"""The retrieval engine, driven deterministically.

Every model call is mocked, so these run with no Ollama and therefore run in CI —
which matters, because the engine's only other exercise is test_retrieval.py, and
that needs a live model and skips without one. Before this file, the two-phase
prune/retrieve loop at the heart of the product was covered at 27%.

What is pinned here is behaviour, not lines: which failures are safe to guess at
and which are not, that pruning never silently loses a branch, and that a
document's own order survives to the answer.
"""

import json

import pytest
from unittest.mock import patch

from pageindex import QuestionRun
from pageindex.document_index import DocumentIndex
from pageindex import search as pi_search
from pageindex.settings import settings
from pageindex.nodes import (
    PageNode,
    _build_nodes_by_id,
    _build_parent_map,
)

KEEP = '{"relevant": true, "reason": "covers it"}'
PRUNE = '{"relevant": false, "reason": "off topic"}'


def leaf(node_id: str, content: str) -> PageNode:
    return PageNode(node_id=node_id, title=node_id.replace("-", " ").title(),
                    heading_level=3, line_idx=1, summary="s", content=content)


def section(node_id: str, children: list[PageNode]) -> PageNode:
    return PageNode(node_id=node_id, title=node_id.replace("-", " ").title(),
                    heading_level=2, line_idx=0, summary="s", children=children)


class TestDescendantOutline:
    def test_sections_and_leaves_are_marked_differently(self):
        tree = section("isolation", [leaf("mrsa", "x"), section("vre", [leaf("vre-room", "y")])])
        outline = pi_search._format_descendant_outline(tree)
        assert "- Mrsa" in outline        # a leaf
        assert "• Vre" in outline         # a section, so the model knows to look deeper

    def test_depth_is_indented_so_the_shape_is_legible(self):
        tree = section("a", [section("b", [leaf("c", "x")])])
        assert "  - C" in pi_search._format_descendant_outline(tree)

    def test_a_childless_node_outlines_to_nothing(self):
        assert pi_search._format_descendant_outline(leaf("solo", "x")) == ""


class TestSectionCheck:
    """Phase 1 decides which branches are even considered. A wrong prune is
    invisible downstream: the leaves under it are never evaluated at all.
    """

    def test_a_keep_verdict_is_recorded(self):
        run = QuestionRun()
        with patch.object(pi_search, "_chat", return_value=KEEP):
            verdict, reason = pi_search._check_section_relevant(
                section("isolation", [leaf("mrsa", "x")]), "q", "crumb", run=run)
        assert verdict is True and reason == "covers it"
        assert "isolation" in run.events()["kept"]

    def test_a_prune_verdict_is_honoured(self):
        with patch.object(pi_search, "_chat", return_value=PRUNE):
            verdict, reason = pi_search._check_section_relevant(
                section("catering", [leaf("menus", "x")]), "q", "crumb", run=QuestionRun())
        assert verdict is False and reason == "off topic"

    def test_an_unreachable_model_keeps_the_section(self):
        """Fails open, deliberately: pruning on a failure would drop a branch
        the model never actually judged, and nothing downstream could tell."""
        with patch.object(pi_search, "_chat", side_effect=RuntimeError("connection refused")):
            verdict, reason = pi_search._check_section_relevant(
                section("isolation", [leaf("mrsa", "x")]), "q", "crumb", run=QuestionRun())
        assert verdict is True
        assert "connection refused" in reason

    def test_an_unparseable_answer_also_keeps_the_section(self):
        # The parser raises rather than returning None, so this lands in the same
        # fail-open path as an outage. Pinned because it is easy to "fix" the
        # parser into returning None and silently invert this.
        with patch.object(pi_search, "_chat", return_value="I'm afraid I can't do that"):
            verdict, _ = pi_search._check_section_relevant(
                section("isolation", [leaf("mrsa", "x")]), "q", "crumb", run=QuestionRun())
        assert verdict is True


class TestExplainNonselection:
    def test_it_reads_the_section_s_own_content(self):
        node = leaf("mrsa", "Contact precautions apply.")
        with patch.object(pi_search, "_chat",
                          return_value='{"topic": "PPE", "addresses_query": false, "reason": "no"}') as chat:
            out = pi_search.explain_nonselection(node, "q", "hygiene_doc", "crumb")
        assert out == {"topic": "PPE", "addresses_query": False, "reason": "no"}
        assert "Contact precautions apply." in chat.call_args[0][0]

    def test_a_section_with_no_text_borrows_its_leaves(self):
        """Judging a heading by its title alone is a guess; the leaves are the
        actual content the explanation has to be anchored to."""
        node = section("isolation", [leaf("mrsa", "Single room required.")])
        with patch.object(pi_search, "_chat",
                          return_value='{"topic": "t", "addresses_query": false, "reason": "r"}') as chat:
            pi_search.explain_nonselection(node, "q", "doc", "crumb")
        assert "Single room required." in chat.call_args[0][0]

    def test_a_failure_explains_nothing_rather_than_inventing(self):
        with patch.object(pi_search, "_chat", side_effect=RuntimeError("down")):
            out = pi_search.explain_nonselection(leaf("mrsa", "x"), "q", "doc", "crumb")
        assert out == {"topic": "", "addresses_query": False, "reason": ""}


class TestPruneAndCollect:
    """BFS pruning. The invariant that matters is bookkeeping: a pruned branch's
    leaves must still be accounted for, or the progress bar never completes and
    the verdict map has holes where a clinician expects an answer.
    """

    @pytest.fixture
    def tree(self):
        return [
            section("isolation", [leaf("mrsa", "Single room."), leaf("vre", "Cohort.")]),
            section("catering", [leaf("menus", "Soup.")]),
        ]

    def _run_prune(self, tree, chat_impl, run):
        nodes_by_id = _build_nodes_by_id(tree)
        parent_map = _build_parent_map(tree)
        node_meta: dict = {}
        with patch.object(pi_search, "_chat", side_effect=chat_impl):
            leaves = pi_search._prune_and_collect(
                tree, "q", parent_map, nodes_by_id, {}, node_meta, run)
        return leaves, node_meta

    def test_a_kept_branch_yields_its_leaves_and_a_pruned_one_does_not(self, tree):
        run = QuestionRun()
        run.set_total(3)
        surviving, meta = self._run_prune(
            tree, lambda prompt, *a: PRUNE if "Catering" in prompt else KEEP, run)

        assert sorted(n.node_id for n in surviving) == ["mrsa", "vre"]
        assert meta["catering"]["status"] == "pruned"
        assert meta["isolation"]["status"] == "kept"

    def test_leaves_under_a_pruned_branch_are_accounted_for(self, tree):
        """Not marking them would leave the run permanently short of its total."""
        run = QuestionRun()
        run.set_total(3)
        _, meta = self._run_prune(
            tree, lambda prompt, *a: PRUNE if "Catering" in prompt else KEEP, run)

        assert meta["menus"]["status"] == "pruned"
        assert "menus" in run.events()["pruned"]
        assert run.progress()["done"] == 1, "the one pruned leaf should count as done"

    def test_a_pruned_leaf_says_why_and_names_its_ancestor(self, tree):
        run = QuestionRun()
        run.set_total(3)
        _, meta = self._run_prune(
            tree, lambda prompt, *a: PRUNE if "Catering" in prompt else KEEP, run)
        assert "Catering" in meta["menus"]["reason"]

    def test_an_outage_prunes_nothing(self, tree):
        run = QuestionRun()
        run.set_total(3)

        def down(*_a, **_k):
            raise RuntimeError("connection refused")

        surviving, _ = self._run_prune(tree, down, run)
        # Every leaf survives to evaluation: the model judged no section, so no
        # branch may be dropped on its behalf.
        assert sorted(n.node_id for n in surviving) == ["menus", "mrsa", "vre"]


class MutableDocumentIndex:
    def __init__(self):
        self.index = None

    @property
    def nodes(self):
        return self.index.nodes


class TestRetrieveWithMetadata:
    @pytest.fixture
    def index(self, tmp_path):
        return MutableDocumentIndex()

    def _write(self, index, root: PageNode):
        index.index = DocumentIndex.from_nodes((root,))

    def test_a_direct_descendant_heading_match_survives_a_false_model_prune(
        self, index
    ):
        root = PageNode(
            node_id="hypertension",
            title="Hypertension",
            heading_level=1,
            line_idx=0,
            summary="s",
            children=[section(
                "non-pharmacological-management",
                [leaf(
                    "sodium-restriction",
                    "Limit sodium intake to less than 2,300 mg per day.",
                )],
            )],
        )
        self._write(index, root)

        def model_verdict(prompt, *_args):
            if "--- SECTION CONTENT ---" in prompt:
                return json.dumps({
                    "relevant": True,
                    "reason": "direct answer",
                    "quote": "Limit sodium intake to less than 2,300 mg per day.",
                })
            return PRUNE

        with patch.object(pi_search, "_chat", side_effect=model_verdict):
            nodes, _ = pi_search.retrieve_with_metadata(
                "guideline",
                "What sodium intake is recommended?",
                QuestionRun(),
                index,
            )

        assert [node.node_id for node in nodes] == ["sodium-restriction"]

    def test_a_leaf_can_answer_one_explicit_part_of_a_compound_query(
        self, index
    ):
        quote = (
            "Train-of-four stimulation is used to titrate neuromuscular "
            "blockade."
        )
        root = PageNode(
            node_id="neuromuscular-blockade",
            title="Neuromuscular blockade",
            heading_level=1,
            line_idx=0,
            summary="ARDS indications and monitoring",
            children=[leaf("monitoring-and-safety", quote)],
        )
        self._write(index, root)

        def model_verdict(prompt, *_args):
            if "--- SECTION CONTENT ---" not in prompt:
                return KEEP
            if "answers at least one of them" not in prompt:
                return json.dumps({"relevant": False})
            return json.dumps({
                "relevant": True,
                "reason": "directly answers the monitoring part",
                "quote": quote,
            })

        with patch.object(pi_search, "_chat", side_effect=model_verdict):
            nodes, _ = pi_search.retrieve_with_metadata(
                "guideline",
                (
                    "When are neuromuscular blocking agents indicated in ARDS "
                    "and how is the depth of blockade monitored?"
                ),
                QuestionRun(),
                index,
            )

        assert [node.node_id for node in nodes] == ["monitoring-and-safety"]

    def test_the_passed_document_index_is_the_document_read(self, index):
        self._write(index, leaf("pinned", "Pinned passage."))
        with patch.object(
            pi_search,
            "_chat",
            return_value=(
                '{"relevant": true, "reason": "exact", '
                '"quote": "Pinned passage."}'
            ),
        ):
            nodes, _ = pi_search.retrieve_with_metadata(
                "doc", "q", QuestionRun(), index
            )
        assert [node.node_id for node in nodes] == ["pinned"]

    def test_an_empty_index_returns_nothing_without_calling_the_model(self, index):
        with pytest.raises(ValueError, match="non-empty list"):
            DocumentIndex.from_serialized("[]")

    def test_a_childless_root_is_itself_a_leaf_and_gets_evaluated(self, index):
        # Worth pinning: a single-section document is not an empty one, and its
        # own text must still be judged rather than skipped for having no children.
        self._write(index, leaf("solo", "Disinfect hands."))
        with patch.object(pi_search, "_chat",
                          return_value='{"relevant": true, "reason": "r", "quote": "Disinfect hands."}'):
            nodes, _ = pi_search.retrieve_with_metadata(
                "doc", "q", QuestionRun(), index
            )
        assert [n.node_id for n in nodes] == ["solo"]

    def test_selected_leaves_keep_the_document_s_own_order(self, index):
        """Sources are numbered [1], [2]… in the answer. If retrieval returned
        them in completion order, the citation numbers would shuffle per run for
        the same question."""
        root = PageNode(
            node_id="root", title="Guideline", heading_level=1, line_idx=0, summary="s",
            children=[leaf("first", "A."), leaf("second", "B."), leaf("third", "C.")],
        )
        self._write(index, root)
        def exact_verdict(prompt, *_args):
            quote = next(value for value in ("A.", "B.", "C.") if value in prompt)
            return json.dumps({"relevant": True, "reason": "r", "quote": quote})

        with patch.object(pi_search, "_chat", side_effect=exact_verdict):
            nodes, _ = pi_search.retrieve_with_metadata(
                "doc", "q", QuestionRun(), index
            )
        assert [n.node_id for n in nodes] == ["first", "second", "third"]

    def test_a_rejected_leaf_is_left_out_of_the_result(self, index):
        root = PageNode(
            node_id="root", title="Guideline", heading_level=1, line_idx=0, summary="s",
            children=[leaf("wanted", "Yes."), leaf("unwanted", "No.")],
        )
        self._write(index, root)

        def verdict(prompt, *_a):
            if "Wanted" in prompt:
                return '{"relevant": true, "reason": "r", "quote": "Yes."}'
            return PRUNE

        with patch.object(pi_search, "_chat", side_effect=verdict):
            nodes, meta = pi_search.retrieve_with_metadata(
                "doc", "q", QuestionRun(), index
            )
        assert [n.node_id for n in nodes] == ["wanted"]
        assert meta["unwanted"]["relevant"] is False

    def test_a_leaf_with_a_fabricated_quote_is_not_retrieved(self, index):
        self._write(index, leaf("isolation", "Use a single room."))
        with patch.object(
            pi_search,
            "_chat",
            return_value=(
                '{"relevant": true, "reason": "claims isolation", '
                '"quote": "Use a negative-pressure room."}'
            ),
        ):
            nodes, meta = pi_search.retrieve_with_metadata(
                "doc", "q", QuestionRun(), index
            )

        assert nodes == []
        assert meta["isolation"]["status"] == "error"

    def test_progress_counts_every_leaf_exactly_once(self, index):
        root = PageNode(
            node_id="root", title="Guideline", heading_level=1, line_idx=0, summary="s",
            children=[leaf(f"n{i}", "text") for i in range(5)],
        )
        self._write(index, root)
        run = QuestionRun()
        run.set_total(5)
        with patch.object(pi_search, "_chat", return_value=PRUNE):
            pi_search.retrieve_with_metadata(
                "doc", "q", run, index
            )
        assert run.progress() == {"total": 5, "done": 5}

    def test_a_non_object_section_reply_fails_open_and_is_recorded(self, index):
        root = PageNode(
            node_id="root",
            title="Guideline",
            heading_level=1,
            line_idx=0,
            summary="s",
            children=[section("isolation", [leaf("mrsa", "Single room.")])],
        )
        self._write(index, root)

        replies = iter(
            [
                '["not", "a", "verdict"]',
                '{"relevant": true, "reason": "states isolation", '
                '"quote": "Single room."}',
            ]
        )
        with patch.object(pi_search, "_chat", side_effect=lambda *_a: next(replies)):
            nodes, meta = pi_search.retrieve_with_metadata(
                "doc", "q", QuestionRun(), index
            )

        assert [node.node_id for node in nodes] == ["mrsa"]
        assert meta["isolation"]["status"] == "error"

    def test_one_retrieval_run_uses_one_model_snapshot(self, index, monkeypatch):
        root = PageNode(
            node_id="root",
            title="Guideline",
            heading_level=1,
            line_idx=0,
            summary="s",
            children=[section("isolation", [leaf("mrsa", "Single room.")])],
        )
        self._write(index, root)
        monkeypatch.setattr(settings, "model", "model-a")
        used_models = []

        def reply(prompt, *_args):
            used_models.append(_args[-1])
            settings.model = "model-b"
            if "Single room." in prompt:
                return (
                    '{"relevant": true, "reason": "exact", '
                    '"quote": "Single room."}'
                )
            return KEEP

        with patch.object(pi_search, "_chat", side_effect=reply):
            pi_search.retrieve_with_metadata(
                "doc", "q", QuestionRun(), index
            )

        assert used_models == ["model-a", "model-a"]
