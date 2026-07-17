"""The CLI consumes the same deep modules as practitioner chat."""

from unittest.mock import MagicMock, patch

import pytest

import pipeline
from api.question_answering import (
    AnswerKind,
    AnswerOutcome,
    EvidenceCitation,
    GroundedAnswer,
)
from api.retrieval import (
    DocumentSearch,
    LibrarySearchResult,
    LibraryStatus,
    SearchDiagnostic,
    VerifiedEvidence,
)
from pageindex.library import ExpectedLibraryStore, LibraryCandidate
from pageindex.nodes import PageNode
from pageindex.pins import VisualLocation

GENERATION_ID = "a" * 64


@pytest.fixture
def ingest_adapters(monkeypatch):
    monkeypatch.setattr(pipeline, "defaults", lambda: {"ingest": "basic_markdown"})
    adapters = {}

    def fake_load(stage, name):
        assert stage == "ingest"
        return adapters.setdefault(name, MagicMock())

    monkeypatch.setattr(pipeline, "load", fake_load)
    return adapters


@pytest.fixture
def kb(tmp_path, monkeypatch):
    directory = tmp_path / "kb"
    directory.mkdir()
    monkeypatch.setattr(pipeline, "KB_DIR", directory)
    return directory


@pytest.fixture
def library_dir(tmp_path, monkeypatch):
    directory = tmp_path / "library"
    directory.mkdir()
    monkeypatch.setattr(pipeline, "LIBRARY_DIR", directory)
    return directory


def search(status=LibraryStatus.COMPLETE, evidence=()):
    searched_document = evidence[0].document if evidence else "ready"
    if status is LibraryStatus.UNAVAILABLE:
        documents = ()
        generation_id = None
    elif status is LibraryStatus.PARTIAL:
        diagnostic = SearchDiagnostic(
            "unavailable", "document_unavailable", "technical"
        )
        documents = (
            DocumentSearch.from_verified_evidence(searched_document, tuple(evidence)),
            DocumentSearch(
                "unavailable",
                (),
                (),
                (diagnostic,),
            ),
        )
        generation_id = GENERATION_ID
    else:
        documents = (
            DocumentSearch.from_verified_evidence(searched_document, tuple(evidence)),
        )
        generation_id = GENERATION_ID
    return LibrarySearchResult(
        query="q",
        generation_id=generation_id,
        documents=documents,
    )


def answer_with(outcome):
    def answer(question, run):
        assert question.text == "q"
        return outcome
    return answer


def citation(evidence: VerifiedEvidence) -> EvidenceCitation:
    return EvidenceCitation(
        id="s1",
        number=1,
        document=evidence.document,
        node_id=evidence.node.node_id,
        title=evidence.node.title,
        breadcrumb=evidence.breadcrumb,
        excerpt=evidence.node.content or "",
        quote=evidence.quote,
        reason=evidence.reason,
        source_href=None,
        visual=VisualLocation("unavailable", reason="missing_provenance_pin"),
    )


def ready_library(library_dir):
    ExpectedLibraryStore(library_dir).publish({
        "ready": LibraryCandidate("# Ready\n\nready\n"),
    })


class TestIngest:
    def test_it_runs_the_default_adapter(self, ingest_adapters):
        pipeline.ingest()
        ingest_adapters["basic_markdown"].run.assert_called_once()

    def test_an_explicit_adapter_wins(self, ingest_adapters):
        pipeline.ingest("betteringest_pdf")
        ingest_adapters["betteringest_pdf"].run.assert_called_once()


class TestIndex:
    def test_every_build_promotes_one_complete_corpus_generation(
        self, kb, monkeypatch
    ):
        for stem in ("a", "b", "c"):
            (kb / f"{stem}.md").write_text("# x", encoding="utf-8")
        build = MagicMock()
        monkeypatch.setattr(pipeline.pageindex, "build_generation", build)

        pipeline.index()

        build.assert_called_once_with(["a", "b", "c"])

    def test_doc_does_not_create_a_partial_generation(self, kb, monkeypatch):
        for stem in ("a", "b"):
            (kb / f"{stem}.md").write_text("# x", encoding="utf-8")
        build = MagicMock()
        monkeypatch.setattr(pipeline.pageindex, "build_generation", build)

        pipeline.index(doc="a")

        build.assert_called_once_with(["a", "b"])

    def test_an_unknown_document_refuses_to_rebuild(self, kb, monkeypatch):
        (kb / "a.md").write_text("# x", encoding="utf-8")
        build = MagicMock()
        monkeypatch.setattr(pipeline.pageindex, "build_generation", build)

        with pytest.raises(SystemExit):
            pipeline.index(doc="missing")

        build.assert_not_called()

    def test_an_empty_knowledge_base_refuses_to_report_success(self, kb, capsys):
        with pytest.raises(SystemExit):
            pipeline.index()
        assert "Run 'pipeline.py ingest' first" in capsys.readouterr().out


class TestQuery:
    def test_an_empty_index_refuses_to_answer(
        self, library_dir, capsys, monkeypatch
    ):
        diagnostic = SearchDiagnostic(
            "__expected_library__",
            "expected_library_unavailable",
            "technical",
        )
        outcome = AnswerOutcome.retrieval_unavailable(
            LibrarySearchResult(
                "q", None, (), (diagnostic,)
            ),
        )
        calls = []

        def answer(question, run):
            calls.append(run.id)
            return outcome

        monkeypatch.setattr(pipeline, "answer_question", answer)
        with pytest.raises(SystemExit):
            pipeline.query("q")
        assert calls == ["cli"]
        assert "Run 'pipeline.py index' first" in capsys.readouterr().out

    def test_it_returns_the_one_grounded_answer(self, library_dir, monkeypatch):
        ready_library(library_dir)
        node = PageNode(
            node_id="ready",
            title="Ready",
            heading_level=1,
            line_idx=0,
            summary="",
            content="ready",
        )
        evidence = VerifiedEvidence(
            "ready", node, "Ready", "exact", "ready"
        )
        outcome = AnswerOutcome.answered(
            search(evidence=(evidence,)),
            GroundedAnswer("Die Antwort [1]."),
            (citation(evidence),),
        )
        monkeypatch.setattr(
            pipeline,
            "answer_question",
            answer_with(outcome),
        )

        assert pipeline.query("q") == "Die Antwort [1]."

    def test_a_complete_negative_is_the_only_honest_empty_result(
        self, library_dir, monkeypatch
    ):
        ready_library(library_dir)
        outcome = AnswerOutcome.insufficient_evidence(search())
        monkeypatch.setattr(
            pipeline,
            "answer_question",
            answer_with(outcome),
        )

        assert pipeline.query("q") == "insufficient_evidence"

    @pytest.mark.parametrize(
        ("kind", "status"),
        [
            (AnswerKind.SEARCH_INCOMPLETE, LibraryStatus.PARTIAL),
            (AnswerKind.RETRIEVAL_UNAVAILABLE, LibraryStatus.UNAVAILABLE),
        ],
    )
    def test_an_incomplete_search_is_never_rephrased_as_no_evidence(
        self, library_dir, monkeypatch, kind, status
    ):
        ready_library(library_dir)
        outcome = (
            AnswerOutcome.search_incomplete(search(status))
            if kind is AnswerKind.SEARCH_INCOMPLETE
            else AnswerOutcome.retrieval_unavailable(search(status))
        )
        monkeypatch.setattr(
            pipeline,
            "answer_question",
            answer_with(outcome),
        )

        with pytest.raises(RuntimeError, match=kind.value):
            pipeline.query("q")

    def test_it_reports_only_verified_evidence(self, library_dir, monkeypatch, capsys):
        ready_library(library_dir)
        node = PageNode(
            node_id="mrsa",
            title="MRSA",
            heading_level=2,
            line_idx=1,
            summary="s",
            content="Einzelzimmer.",
        )
        evidence = VerifiedEvidence(
            "hygiene", node, "Hygiene › MRSA", "relevant", "Einzelzimmer."
        )
        outcome = AnswerOutcome.answered(
            search(evidence=(evidence,)),
            GroundedAnswer("Einzelzimmer [1]."),
            (citation(evidence),),
        )
        monkeypatch.setattr(
            pipeline,
            "answer_question",
            answer_with(outcome),
        )

        pipeline.query("q")

        assert "hygiene / mrsa" in capsys.readouterr().out


class TestResolveQuery:
    def test_it_takes_the_argument_when_given(self):
        assert pipeline._resolve_query("  MRSA?  ") == "MRSA?"

    def test_it_asks_when_omitted(self):
        with patch("builtins.input", return_value="typed question"):
            assert pipeline._resolve_query(None) == "typed question"

    def test_an_empty_query_exits(self):
        with patch("builtins.input", return_value="   "):
            with pytest.raises(SystemExit):
                pipeline._resolve_query("")


def test_module_listing_contains_only_real_ingest_adapters(
    ingest_adapters, capsys
):
    del ingest_adapters
    with patch.object(pipeline, "discover", return_value={
        "ingest": {
            "basic_markdown": {"label": "Basic", "description": "d"},
            "betteringest_pdf": {"label": "Better", "description": "d"},
        },
    }):
        pipeline.cmd_list(None)
    output = capsys.readouterr().out
    assert " * basic_markdown" in output
    assert "   betteringest_pdf" in output
    assert "Index" not in output and "Query" not in output


def test_no_command_prints_help(capsys):
    with patch.object(pipeline.sys, "argv", ["pipeline.py"]):
        pipeline.main()
    assert "usage:" in capsys.readouterr().out
