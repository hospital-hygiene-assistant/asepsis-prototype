"""Running one retrieval pass across every indexed document.

The layer /api/chat and /api/run both drive. It decides what the answer is
allowed to be built from, so what it quietly drops matters as much as what it
returns.
"""

import json

import pytest

from api import retrieval as api_retrieval
from api.retrieval import count_eval_errors, run_retrieval
from api.runs import RunRegistry
from pageindex.nodes import PageNode, _node_to_dict


def leaf(node_id: str, content: str = "text") -> PageNode:
    return PageNode(node_id=node_id, title=node_id.title(), heading_level=2,
                    line_idx=1, summary="s", content=content)


def doc_tree(*leaves: PageNode) -> PageNode:
    return PageNode(node_id="root", title="Guideline", heading_level=1, line_idx=0,
                    summary="s", children=list(leaves))


class FakeIndexModule:
    """Stands in for modules/index/*: run_retrieval duck-types this surface."""

    def __init__(self, per_doc: dict):
        self._per_doc = per_doc

    def retrieve_with_metadata(self, doc_name, query, state):
        return self._per_doc[doc_name]

    def retrieve(self, doc_name, query, state):
        return self._per_doc[doc_name][0]


class BareIndexModule:
    """Only the plain retrieve(). run_retrieval getattr-checks for the richer
    surface, so a plugin offering just this must still work."""

    def __init__(self, per_doc: dict):
        self._per_doc = per_doc

    def retrieve(self, doc_name, query, state):
        return self._per_doc[doc_name][0]


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api_retrieval, "INDEX_DIR", tmp_path)
    return tmp_path


def write_index(index_dir, stem: str, root: PageNode):
    (index_dir / f"{stem}.json").write_text(
        json.dumps([_node_to_dict(root)]), encoding="utf-8")


@pytest.fixture
def run():
    return RunRegistry().create("probe")


class TestRunRetrieval:
    def test_it_gathers_every_document(self, index_dir, run):
        write_index(index_dir, "hygiene", doc_tree(leaf("mrsa")))
        write_index(index_dir, "antibiotics", doc_tree(leaf("dosing")))
        mod = FakeIndexModule({
            "hygiene": ([leaf("mrsa")], {"mrsa": {"reason": "r", "quote": "q"}}),
            "antibiotics": ([], {}),
        })
        results = run_retrieval("q", mod, run)
        assert sorted(results) == ["antibiotics", "hygiene"]

    def test_a_retrieved_node_carries_its_reason_and_quote(self, index_dir, run):
        """These are what the clinician is shown next to the citation; losing
        them here turns a justified source into an unexplained one."""
        write_index(index_dir, "hygiene", doc_tree(leaf("mrsa", "Single room.")))
        mod = FakeIndexModule({
            "hygiene": ([leaf("mrsa", "Single room.")],
                        {"mrsa": {"reason": "states isolation", "quote": "Single room."}}),
        })
        node = run_retrieval("q", mod, run)["hygiene"]["nodes"][0]
        assert node["reason"] == "states isolation"
        assert node["quote"] == "Single room."
        assert node["content"] == "Single room."

    def test_progress_is_totalled_before_any_leaf_is_evaluated(self, index_dir, run):
        """The client polls from the moment it sends the question, so the
        denominator has to exist before the first verdict, not after."""
        write_index(index_dir, "a", doc_tree(leaf("l1"), leaf("l2")))
        write_index(index_dir, "b", doc_tree(leaf("l3")))
        mod = FakeIndexModule({"a": ([], {}), "b": ([], {})})
        run_retrieval("q", mod, run)
        assert run.state.progress()["total"] == 3

    def test_a_module_without_metadata_still_works(self, index_dir, run):
        # The index stage is a plugin; only the richer surface is optional.
        write_index(index_dir, "hygiene", doc_tree(leaf("mrsa")))
        mod = BareIndexModule({"hygiene": ([leaf("mrsa")], {})})
        results = run_retrieval("q", mod, run)
        assert results["hygiene"]["retrieved_ids"] == ["mrsa"]
        assert results["hygiene"]["nodes"][0]["reason"] == ""

    def test_an_empty_corpus_is_an_empty_result_not_a_crash(self, index_dir, run):
        assert run_retrieval("q", FakeIndexModule({}), run) == {}


class TestCountEvalErrors:
    def test_it_counts_only_what_was_never_checked(self):
        results = {
            "a": {"node_meta": {"n1": {"status": "error"}, "n2": {"status": "rejected"}}},
            "b": {"node_meta": {"n3": {"status": "error"}, "n4": {"status": "retrieved"}}},
        }
        assert count_eval_errors(results) == 2

    def test_a_pruned_leaf_is_not_an_error(self):
        # Pruning is a decision the model made; an error is one it never made.
        results = {"a": {"node_meta": {"n1": {"status": "pruned"}, "n2": {"status": "kept"}}}}
        assert count_eval_errors(results) == 0
