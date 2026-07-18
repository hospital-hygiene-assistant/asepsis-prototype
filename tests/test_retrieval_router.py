"""Debug retrieval routes read the immutable Expected library."""

import time

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
        "hygiene": LibraryCandidate("# Other\n\nWear a gown.\n"),
    })

    monkeypatch.setattr(retrieval_router, "LIBRARY_DIR", library)
    monkeypatch.setattr(retrieval_router, "ensure_explainer", lambda: None)

    response = retrieval_router.explain_node(retrieval_router.ExplainRequest(
        generation_id=first.generation_id,
        stem="hygiene",
        node_id="ppe",
        query="q",
    ))

    assert response.status_code == 503


def test_debug_results_carry_generation_scoped_reader_links(monkeypatch):
    generation_id = "a" * 64

    class Search:
        status = type("Status", (), {"value": "complete"})()
        diagnostics = ()

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
    monkeypatch.setattr(
        retrieval_router,
        "_debug_results",
        lambda _search: {"hygiene guide": {"tree": [], "nodes": []}},
    )

    response = retrieval_router.run_query(
        retrieval_router.RunRequest(query="q")
    )
    run_id = response.body.decode("utf-8").split('"run_id":"', 1)[1].split('"', 1)[0]
    from api.runs import registry
    deadline = time.monotonic() + 1
    while (snapshot := registry.snapshot(run_id))["phase"] != "completed":
        assert time.monotonic() < deadline
        time.sleep(0.005)
    document = str(snapshot["result"])

    assert (
        f"/api/library/{generation_id}/documents/hygiene%20guide/full"
        in document
    )
    assert (
        f"/api/library/{generation_id}/documents/hygiene%20guide/page/{{page}}"
        in document
    )
