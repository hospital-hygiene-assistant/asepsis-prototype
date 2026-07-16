"""Immutable corpus generations and atomic promotion."""

import json

from api.retrieval import WholeLibraryRetrieval
from api.runs import RunRegistry
from api.routers import documents as documents_router
from api.routers import retrieval as retrieval_router
from pageindex.generations import IndexGenerationStore
from pageindex.nodes import PageNode, _node_to_dict


def serialised_leaf(node_id: str, content: str) -> tuple[str, PageNode]:
    node = PageNode(
        node_id=node_id,
        title=node_id.title(),
        heading_level=1,
        line_idx=0,
        summary="s",
        content=content,
    )
    return json.dumps([_node_to_dict(node)]), node


def test_publish_writes_a_complete_generation_before_pointing_current_at_it(tmp_path):
    store = IndexGenerationStore(tmp_path)

    published = store.publish(
        {
            "hygiene": '[{"nodeId": "hygiene"}]',
            "isolation": '[{"nodeId": "isolation"}]',
        }
    )

    pointer = json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))
    generation_dir = tmp_path / "generations" / pointer["generation_id"]
    manifest = json.loads((generation_dir / "manifest.json").read_text(encoding="utf-8"))
    assert pointer["generation_id"] == published.generation_id
    assert manifest["documents"] == ["hygiene", "isolation"]
    assert sorted(path.stem for path in published.document_paths) == [
        "hygiene",
        "isolation",
    ]
    assert all(path.parent == generation_dir for path in published.document_paths)


def test_publish_rejects_a_document_without_searchable_leaves(tmp_path):
    store = IndexGenerationStore(tmp_path)

    try:
        store.publish({"empty": "[]"})
    except ValueError as exc:
        assert "no searchable leaves" in str(exc)
    else:
        raise AssertionError("an empty document tree was published")


def test_a_search_keeps_one_generation_when_current_changes_mid_run(tmp_path):
    store = IndexGenerationStore(tmp_path)
    first_json, first = serialised_leaf("first", "First exact passage.")
    second_json, second = serialised_leaf("second", "Second exact passage.")
    initial = store.publish({"a": first_json, "b": second_json})

    class PromoteDuringSearch:
        def __init__(self):
            self.paths = []

        def retrieve_with_metadata_from_path(
            self, document, query, state, index_path, model=None
        ):
            self.paths.append(index_path)
            if len(self.paths) == 1:
                replacement, _ = serialised_leaf("replacement", "Replacement.")
                store.publish({"replacement": replacement})
            node = first if document == "a" else second
            return [node], {
                node.node_id: {
                    "status": "retrieved",
                    "reason": "exact",
                    "quote": node.content,
                }
            }

    index = PromoteDuringSearch()
    result = WholeLibraryRetrieval(index, index_dir=tmp_path).search(
        "question", RunRegistry().create("probe")
    )

    assert result.generation_id == initial.generation_id
    assert [path.parent.name for path in index.paths] == [
        initial.generation_id,
        initial.generation_id,
    ]
    assert [item.document for item in result.evidence] == ["a", "b"]


def test_the_document_listing_reads_the_promoted_generation(tmp_path, monkeypatch):
    payload, _ = serialised_leaf("current", "Current passage.")
    IndexGenerationStore(tmp_path).publish({"current_doc": payload})
    (tmp_path / "stale_flat.json").write_text(payload, encoding="utf-8")
    monkeypatch.setattr(documents_router, "INDEX_DIR", tmp_path)

    response = documents_router.get_documents()
    listed = json.loads(response.body)

    assert [document["name"] for document in listed] == ["current_doc"]


def test_explain_resolves_the_current_generation_not_a_flat_stale_file(
    tmp_path, monkeypatch
):
    payload, _ = serialised_leaf("current", "Current passage.")
    IndexGenerationStore(tmp_path).publish({"current_doc": payload})
    monkeypatch.setattr(retrieval_router, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(retrieval_router, "ensure_explainer", lambda: None)

    response = retrieval_router.explain_node(retrieval_router.ExplainRequest(
        stem="current_doc", node_id="current", query="q"
    ))

    assert response.status_code == 503


def test_publish_never_deletes_a_generation_another_process_may_be_reading(tmp_path):
    store = IndexGenerationStore(tmp_path)
    one, _ = serialised_leaf("one", "One.")
    two, _ = serialised_leaf("two", "Two.")
    three, _ = serialised_leaf("three", "Three.")
    four, _ = serialised_leaf("four", "Four.")
    first = store.publish({"doc": one})

    store.publish({"doc": two})
    store.publish({"doc": three})
    store.publish({"doc": four})
    retained = {
        path.name
        for path in store.generations.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    assert first.generation_id in retained
    assert len(retained) == 4
