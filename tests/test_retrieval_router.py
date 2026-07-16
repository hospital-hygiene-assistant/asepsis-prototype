"""Debug retrieval routes read the immutable Expected library."""

from api.routers import retrieval as retrieval_router
from pageindex.library import ExpectedLibraryStore, LibraryCandidate


def test_explain_reads_the_requested_expected_library_generation(
    tmp_path, monkeypatch
):
    library = tmp_path / "library"
    store = ExpectedLibraryStore(library)
    first = store.publish({
        "hygiene": LibraryCandidate("# PPE\n\nWear gloves.\n"),
    })
    store.publish({
        "hygiene": LibraryCandidate("# PPE\n\nWear a gown.\n"),
    })
    opened = {}
    read_tree = retrieval_router.read_tree

    def capture_generation(index_path):
        opened["index_path"] = index_path
        return read_tree(index_path)

    monkeypatch.setattr(retrieval_router, "LIBRARY_DIR", library)
    monkeypatch.setattr(retrieval_router, "read_tree", capture_generation)
    monkeypatch.setattr(retrieval_router, "ensure_explainer", lambda: None)

    response = retrieval_router.explain_node(retrieval_router.ExplainRequest(
        generation_id=first.generation_id,
        stem="hygiene",
        node_id="ppe",
        query="q",
    ))

    assert response.status_code == 503
    assert first.generation_id in str(opened["index_path"])


def test_debug_results_carry_generation_scoped_reader_links(monkeypatch):
    generation_id = "a" * 64

    class Search:
        status = type("Status", (), {"value": "complete"})()
        diagnostics = ()

        @staticmethod
        def to_debug_results():
            return {"hygiene guide": {"tree": []}}

    Search.generation_id = generation_id

    class Retrieval:
        @staticmethod
        def search(_query, _run):
            return Search()

    monkeypatch.setattr(
        retrieval_router,
        "build_whole_library_retrieval",
        lambda: Retrieval(),
    )

    response = retrieval_router.run_query(
        retrieval_router.RunRequest(query="q")
    )
    document = response.body.decode("utf-8")

    assert (
        f"/api/library/{generation_id}/documents/hygiene%20guide/full"
        in document
    )
    assert (
        f"/api/library/{generation_id}/documents/hygiene%20guide/page/{{page}}"
        in document
    )
