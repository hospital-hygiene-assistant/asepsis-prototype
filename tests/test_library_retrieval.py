"""Whole-library retrieval truth through its public interface."""

from types import SimpleNamespace

import pytest

from api.retrieval import DocumentSearch, LibraryStatus, WholeLibraryRetrieval
from api.routers.retrieval import _debug_results
from pageindex import QuestionRun
from pageindex import DocumentRetrieval, PassageDecision, PassageDecisionKind
from pageindex.library import ExpectedLibraryStore, LibraryCandidate
from pageindex.nodes import PageNode


def leaf(node_id: str, content: str) -> PageNode:
    return PageNode(
        node_id=node_id,
        title=node_id.title(),
        heading_level=2,
        line_idx=1,
        summary="s",
        content=content,
    )


def publish_library(tmp_path, documents: dict[str, tuple[PageNode, ...]]):
    store = ExpectedLibraryStore(tmp_path / "library")
    store.publish({
        document: LibraryCandidate("\n\n".join((
            "# Guideline",
            *(f"## {node.title}\n\n{node.content}" for node in leaves),
        )))
        for document, leaves in documents.items()
    })
    return store


class ScriptedIndex:
    def __init__(self, results: dict):
        self.results = results

    def search_document(
        self, document, query, state, path, model=None
    ):
        _nodes, metadata = self.results[document]
        return scripted_retrieval(document, metadata)


def scripted_retrieval(document, metadata):
    kinds = {
        "retrieved": PassageDecisionKind.PASSAGE_RETRIEVED,
        "rejected": PassageDecisionKind.PASSAGE_REJECTED,
        "error": PassageDecisionKind.SECTION_CHECK_FAILED,
        "kept": PassageDecisionKind.SECTION_KEPT,
        "pruned": PassageDecisionKind.PASSAGE_PRUNED,
    }
    return DocumentRetrieval(
        tuple(
            PassageDecision(
                node_id,
                kinds[meta["status"]],
                str(meta.get("reason") or ""),
                str(meta.get("quote") or ""),
                meta.get("code"),
                document_id=document,
            )
            for node_id, meta in metadata.items()
        ),
    )


def test_a_fully_searched_library_exposes_only_exactly_quoted_evidence(tmp_path):
    passage = leaf("isolation", "MRSA patients require a single room.")
    library = publish_library(tmp_path, {"hygiene": (passage,)})
    index = ScriptedIndex(
        {
            "hygiene": (
                [passage],
                {
                    "isolation": {
                        "status": "retrieved",
                        "reason": "states the measure",
                        "quote": "require a single room",
                    }
                },
            )
        }
    )

    result = WholeLibraryRetrieval(index, library_store=library).search(
        "MRSA?", QuestionRun("probe")
    )

    assert result.status is LibraryStatus.COMPLETE
    assert [(item.document, item.node.node_id, item.quote) for item in result.evidence] == [
        ("hygiene", "isolation", "require a single room")
    ]
    assert result.evidence[0].breadcrumb == "Guideline > Isolation"


def test_a_retrieved_decision_cannot_disappear_from_evidence():
    decision = PassageDecision(
        "isolation",
        PassageDecisionKind.PASSAGE_RETRIEVED,
        "states the measure",
        "single room",
        document_id="hygiene",
    )

    with pytest.raises(ValueError, match="missing evidence or a diagnostic"):
        DocumentSearch("hygiene", (decision,), (), ())


def test_a_passage_decision_cannot_cross_document_searches():
    decision = PassageDecision(
        "isolation",
        PassageDecisionKind.PASSAGE_REJECTED,
        document_id="other-guide",
    )

    with pytest.raises(ValueError, match="different document"):
        DocumentSearch("hygiene", (decision,), (), ())


def test_an_invalid_quote_is_excluded_and_makes_coverage_partial(tmp_path):
    passage = leaf("isolation", "MRSA patients require a single room.")
    library = publish_library(tmp_path, {"hygiene": (passage,)})
    index = ScriptedIndex(
        {
            "hygiene": (
                [passage],
                {
                    "isolation": {
                        "status": "retrieved",
                        "reason": "claims a measure",
                        "quote": "Use a negative-pressure room for 14 days.",
                    }
                },
            )
        }
    )

    result = WholeLibraryRetrieval(index, library_store=library).search(
        "MRSA?", QuestionRun("probe")
    )

    assert result.status is LibraryStatus.PARTIAL
    assert result.evidence == ()
    assert [(item.document, item.node_id, item.code) for item in result.diagnostics] == [
        ("hygiene", "isolation", "invalid_quote")
    ]


def test_a_document_lost_during_search_remains_named_as_partial_coverage(tmp_path):
    present = leaf("present", "Use hand disinfectant.")
    vanished = leaf("vanished", "Use protective equipment.")
    library = publish_library(tmp_path, {
        "a_present": (present,),
        "b_vanished": (vanished,),
    })

    class OneDocumentVanishes:
        def search_document(
            self, document, query, state, path, model=None
        ):
            if document == "b_vanished":
                raise FileNotFoundError("index deleted during retrieval")
            return scripted_retrieval(document, {
                "present": {
                    "status": "retrieved",
                    "reason": "states hygiene",
                    "quote": "hand disinfectant",
                }
            })

    result = WholeLibraryRetrieval(
        OneDocumentVanishes(), library_store=library
    ).search("MRSA?", QuestionRun("probe"))

    assert result.status is LibraryStatus.PARTIAL
    assert [document.document for document in result.documents] == [
        "a_present",
        "b_vanished",
    ]
    missing = result.document("b_vanished")
    assert missing is not None
    assert missing.status.value == "unavailable"
    assert [(item.document, item.code) for item in result.diagnostics] == [
        ("b_vanished", "document_unavailable")
    ]


def test_an_unjudged_section_is_named_and_prevents_complete_coverage(tmp_path):
    passage = leaf("isolation", "Use a single room.")
    library = publish_library(tmp_path, {"hygiene": (passage,)})
    index = ScriptedIndex(
        {
            "hygiene": (
                [passage],
                {
                    "section": {
                        "status": "error",
                        "reason": "the model returned no usable section verdict",
                    },
                    "isolation": {
                        "status": "retrieved",
                        "reason": "states isolation",
                        "quote": "single room",
                    },
                },
            )
        }
    )

    result = WholeLibraryRetrieval(index, library_store=library).search(
        "MRSA?", QuestionRun("probe")
    )

    assert result.status is LibraryStatus.PARTIAL
    assert [item.node.node_id for item in result.evidence] == ["isolation"]
    assert [(item.node_id, item.code) for item in result.diagnostics] == [
        ("section", "unevaluated_node")
    ]


def test_progress_is_totalled_before_any_document_search(tmp_path):
    library = publish_library(tmp_path, {
        "a": (leaf("l1", "one"), leaf("l2", "two")),
        "b": (leaf("l3", "three"),),
    })
    engine = ScriptedIndex({"a": ([], {}), "b": ([], {})})
    run = QuestionRun("probe")

    WholeLibraryRetrieval(engine, library_store=library).search("q", run)

    assert run.progress()["total"] == 3


def test_one_run_uses_one_retrieval_model_across_every_document(tmp_path):
    library = publish_library(tmp_path, {
        "a": (leaf("l1", "one"),),
        "b": (leaf("l2", "two"),),
    })

    class ModelChangingEngine:
        settings = SimpleNamespace(model="model-a")

        def __init__(self):
            self.seen = []

        def search_document(
            self, document, query, state, path, model=None
        ):
            self.seen.append(model)
            self.settings.model = "model-b"
            return scripted_retrieval([], {})

    engine = ModelChangingEngine()

    WholeLibraryRetrieval(engine, library_store=library).search(
        "q", QuestionRun("probe")
    )

    assert engine.seen == ["model-a", "model-a"]


def test_debug_adapter_preserves_verified_reason_quote_and_content(tmp_path):
    passage = leaf("isolation", "Use a single room.")
    library = publish_library(tmp_path, {"hygiene": (passage,)})
    engine = ScriptedIndex({
        "hygiene": ([passage], {
            "isolation": {
                "status": "retrieved",
                "reason": "states isolation",
                "quote": "single room",
            },
        }),
    })

    search = WholeLibraryRetrieval(engine, library_store=library).search(
        "q", QuestionRun("probe")
    )
    result = _debug_results(search, library)

    node = result["hygiene"]["nodes"][0]
    assert node["reason"] == "states isolation"
    assert node["quote"] == "single room"
    assert node["content"] == "Use a single room."


def test_an_unbuilt_expected_library_is_unavailable_not_an_honest_negative(tmp_path):
    result = WholeLibraryRetrieval(
        ScriptedIndex({}),
        library_store=ExpectedLibraryStore(tmp_path / "library"),
    ).search("q", QuestionRun("probe"))

    assert result.status is LibraryStatus.UNAVAILABLE
    assert result.generation_id is None
    assert result.documents == ()
    assert result.diagnostics[0].code == "expected_library_unavailable"
