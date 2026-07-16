"""The CLI consumes the same deep modules as practitioner chat."""

from unittest.mock import MagicMock, patch

import pytest

import pipeline
from api.question_answering import AnswerKind, AnswerOutcome, GroundedAnswer
from api.retrieval import LibrarySearchResult, LibraryStatus, VerifiedEvidence
from pageindex.nodes import PageNode
from pageindex.nodes import _node_to_dict
from pageindex.generations import IndexGenerationStore
import json


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
def index_dir(tmp_path, monkeypatch):
    directory = tmp_path / "index"
    directory.mkdir()
    monkeypatch.setattr(pipeline, "INDEX_DIR", directory)
    return directory


def search(status=LibraryStatus.COMPLETE, evidence=()):
    return LibrarySearchResult(
        query="q",
        generation_id="generation-test",
        status=status,
        documents=(),
        evidence=evidence,
        diagnostics=(),
    )


def answering(outcome):
    class StubAnswering:
        def answer(self, question, run):
            assert question.text == "q"
            return outcome

    return StubAnswering()


def ready_index(index_dir):
    node = PageNode(
        node_id="ready",
        title="Ready",
        heading_level=1,
        line_idx=0,
        summary="s",
        content="ready",
    )
    IndexGenerationStore(index_dir).publish({
        "ready": json.dumps([_node_to_dict(node)]),
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
    def test_an_empty_index_refuses_to_answer(self, index_dir, capsys):
        with pytest.raises(SystemExit):
            pipeline.query("q")
        assert "Run 'pipeline.py index' first" in capsys.readouterr().out

    def test_it_returns_the_one_grounded_answer(self, index_dir, monkeypatch):
        ready_index(index_dir)
        outcome = AnswerOutcome(
            AnswerKind.ANSWERED,
            search(),
            GroundedAnswer("Die Antwort [1]."),
        )
        monkeypatch.setattr(
            pipeline, "_question_answering", lambda _index: answering(outcome)
        )

        assert pipeline.query("q") == "Die Antwort [1]."

    def test_a_complete_negative_is_the_only_honest_empty_result(
        self, index_dir, monkeypatch
    ):
        ready_index(index_dir)
        outcome = AnswerOutcome(AnswerKind.INSUFFICIENT_EVIDENCE, search())
        monkeypatch.setattr(
            pipeline, "_question_answering", lambda _index: answering(outcome)
        )

        result = pipeline.query("q")

        assert "complete library search" in result

    @pytest.mark.parametrize(
        ("kind", "status"),
        [
            (AnswerKind.SEARCH_INCOMPLETE, LibraryStatus.PARTIAL),
            (AnswerKind.RETRIEVAL_UNAVAILABLE, LibraryStatus.UNAVAILABLE),
        ],
    )
    def test_an_incomplete_search_is_never_rephrased_as_no_evidence(
        self, index_dir, monkeypatch, kind, status
    ):
        ready_index(index_dir)
        outcome = AnswerOutcome(kind, search(status))
        monkeypatch.setattr(
            pipeline, "_question_answering", lambda _index: answering(outcome)
        )

        with pytest.raises(RuntimeError, match=kind.value):
            pipeline.query("q")

    def test_it_reports_only_verified_evidence(self, index_dir, monkeypatch, capsys):
        ready_index(index_dir)
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
        outcome = AnswerOutcome(
            AnswerKind.ANSWERED,
            search(evidence=(evidence,)),
            GroundedAnswer("Einzelzimmer [1]."),
        )
        monkeypatch.setattr(
            pipeline, "_question_answering", lambda _index: answering(outcome)
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
