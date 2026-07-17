"""Ingest promotes a complete immutable index generation."""

from unittest.mock import MagicMock

from api.routers import ingest as ingest_router


class ImmediateThread:
    def __init__(self, *, target, daemon):
        self.target = target

    def start(self):
        self.target()


def test_ingest_run_promotes_one_generation_after_the_corpus_is_ready(
    tmp_path, monkeypatch
):
    kb = tmp_path / "kb"
    kb.mkdir()
    for stem in ("a", "b"):
        (kb / f"{stem}.md").write_text("# document", encoding="utf-8")
    ingest_module = MagicMock()
    ingest_module.run.return_value = {}
    build_generation = MagicMock()

    monkeypatch.setattr(ingest_router, "KB_DIR", kb)
    monkeypatch.setattr(ingest_router.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        ingest_router,
        "_load_module",
        lambda stage, name: ingest_module,
    )
    monkeypatch.setattr(ingest_router.pageindex, "build_generation", build_generation)
    ingest_router._ingest_state["state"] = "idle"

    ingest_router.run_ingest(
        ingest_router.IngestRunRequest(ingest_module="basic_markdown")
    )

    build_generation.assert_called_once_with(["a", "b"])


def test_review_ingest_stops_before_publication_and_exposes_the_session(monkeypatch):
    document = MagicMock(doc_name="reviewed-guide")
    ingest_module = MagicMock()
    ingest_module.MODULE_INFO = {"review_required": True, "source": "pdf_folder"}
    ingest_module.prepare_review.return_value = [document]
    review_session = MagicMock(session_id="a" * 32)
    review_session.documents = [MagicMock(document_id="reviewed-guide")]
    review_store = MagicMock()
    review_store.create.return_value = review_session
    build_generation = MagicMock()

    monkeypatch.setattr(ingest_router.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(ingest_router, "_load_module", lambda stage, name: ingest_module)
    monkeypatch.setattr(ingest_router, "review_store", review_store)
    monkeypatch.setattr(ingest_router.pageindex, "build_generation", build_generation)
    ingest_router._ingest_state["state"] = "idle"

    ingest_router.run_ingest(ingest_router.IngestRunRequest(source_dir="/incoming"))

    ingest_module.prepare_review.assert_called_once()
    review_store.create.assert_called_once_with([document])
    build_generation.assert_not_called()
    assert ingest_router._ingest_state["state"] == "review"
    assert ingest_router._ingest_state["review_id"] == "a" * 32
