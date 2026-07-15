"""The CLI: ingest → index → query.

The other half of the product's surface, and the half nothing watched — its only
prior test was that the module imports, at 13% coverage. It is also where the
"No relevant passages found." line lives that /api/chat was fixed for and this
was not (see C3 in test_known_gaps.py).
"""

import pytest
from unittest.mock import MagicMock, patch

import pipeline
from pageindex.nodes import PageNode


@pytest.fixture
def registry(monkeypatch):
    """Stand in for modules/registry: pipeline resolves every stage through it."""
    monkeypatch.setattr(pipeline, "defaults", lambda: {
        "ingest": "basic_markdown", "index": "pageindex_custom", "query": "ollama_synthesis",
    })
    mods = {}

    def fake_load(stage, name):
        return mods.setdefault(f"{stage}/{name}", MagicMock())

    monkeypatch.setattr(pipeline, "load", fake_load)
    return mods


@pytest.fixture
def kb(tmp_path, monkeypatch):
    d = tmp_path / "kb"
    d.mkdir()
    monkeypatch.setattr(pipeline, "KB_DIR", d)
    return d


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    d = tmp_path / "index"
    d.mkdir()
    monkeypatch.setattr(pipeline, "INDEX_DIR", d)
    return d


def leaf(node_id: str) -> PageNode:
    return PageNode(node_id=node_id, title=node_id.title(), heading_level=2,
                    line_idx=1, summary="s", content="text")


class TestIngest:
    def test_it_runs_the_default_module(self, registry):
        pipeline.ingest()
        registry["ingest/basic_markdown"].run.assert_called_once()

    def test_an_explicit_module_wins(self, registry):
        pipeline.ingest("betteringest_pdf")
        registry["ingest/betteringest_pdf"].run.assert_called_once()


class TestIndex:
    def test_a_single_document_is_built_alone(self, registry, kb):
        (kb / "a.md").write_text("# A", encoding="utf-8")
        (kb / "b.md").write_text("# B", encoding="utf-8")
        pipeline.index(doc="a")
        registry["index/pageindex_custom"].build_index.assert_called_once_with("a")

    def test_every_document_is_built_by_default(self, registry, kb):
        for stem in ("a", "b", "c"):
            (kb / f"{stem}.md").write_text("# x", encoding="utf-8")
        pipeline.index()
        built = [c.args[0] for c in registry["index/pageindex_custom"].build_index.call_args_list]
        assert sorted(built) == ["a", "b", "c"]

    def test_an_empty_knowledge_base_exits_rather_than_reporting_success(self, registry, kb, capsys):
        """Building nothing and saying "Done." would leave an empty corpus that
        answers every question with an absence."""
        with pytest.raises(SystemExit) as exit_info:
            pipeline.index()
        assert exit_info.value.code == 1
        assert "Run 'pipeline.py ingest' first" in capsys.readouterr().out


class TestQuery:
    def test_an_empty_index_exits_rather_than_answering(self, registry, index_dir, capsys):
        with pytest.raises(SystemExit) as exit_info:
            pipeline.query("q")
        assert exit_info.value.code == 1
        assert "Run 'pipeline.py index' first" in capsys.readouterr().out

    def test_it_retrieves_across_every_document_and_synthesises(self, registry, index_dir):
        (index_dir / "hygiene.json").write_text("[]", encoding="utf-8")
        (index_dir / "antibiotics.json").write_text("[]", encoding="utf-8")
        index_mod = registry.setdefault("index/pageindex_custom", MagicMock())
        index_mod.retrieve.side_effect = lambda stem, text: [leaf("n1")] if stem == "hygiene" else []
        query_mod = registry.setdefault("query/ollama_synthesis", MagicMock())
        query_mod.synthesise.return_value = "the answer"

        assert pipeline.query("q") == "the answer"
        # Only documents with hits reach synthesis; an empty one is not a source.
        nodes_by_doc = query_mod.synthesise.call_args.args[1]
        assert list(nodes_by_doc) == ["hygiene"]

    def test_it_reports_what_it_retrieved(self, registry, index_dir, capsys):
        (index_dir / "hygiene.json").write_text("[]", encoding="utf-8")
        registry.setdefault("index/pageindex_custom", MagicMock()).retrieve.return_value = [leaf("mrsa")]
        pipeline.query("q")
        out = capsys.readouterr().out
        assert "Retrieved 1 leaf node(s) across 1 document(s)" in out
        assert "hygiene / mrsa" in out

    def test_one_failing_document_does_not_abort_the_others(self, registry, index_dir, capsys):
        """The intent is right — a single bad index should not lose the rest.
        What it must not do is then answer as though the library was fully read;
        that is C3, pinned in test_known_gaps.py."""
        (index_dir / "good.json").write_text("[]", encoding="utf-8")
        (index_dir / "bad.json").write_text("[]", encoding="utf-8")
        index_mod = registry.setdefault("index/pageindex_custom", MagicMock())

        def retrieve(stem, text):
            if stem == "bad":
                raise RuntimeError("index corrupt")
            return [leaf("n1")]

        index_mod.retrieve.side_effect = retrieve
        query_mod = registry.setdefault("query/ollama_synthesis", MagicMock())
        pipeline.query("q")
        assert "Warning: retrieval failed for bad" in capsys.readouterr().out
        assert list(query_mod.synthesise.call_args.args[1]) == ["good"]


class TestResolveQuery:
    def test_it_takes_the_argument_when_given(self):
        assert pipeline._resolve_query("  MRSA?  ") == "MRSA?"

    def test_it_asks_when_omitted(self):
        with patch("builtins.input", return_value="typed question"):
            assert pipeline._resolve_query(None) == "typed question"

    def test_an_empty_query_exits(self, capsys):
        with patch("builtins.input", return_value="   "):
            with pytest.raises(SystemExit) as exit_info:
                pipeline._resolve_query("")
        assert exit_info.value.code == 1
        assert "No query provided." in capsys.readouterr().out


class TestModuleListing:
    def test_the_default_is_marked(self, registry, capsys):
        with patch.object(pipeline, "discover", return_value={
            "ingest": {"basic_markdown": {"label": "Basic", "description": "d"},
                       "betteringest_pdf": {"label": "Better", "description": "d"}},
            "index": {}, "query": {},
        }):
            pipeline.cmd_list(None)
        out = capsys.readouterr().out
        assert " * basic_markdown" in out       # the default, starred
        assert "   betteringest_pdf" in out     # the alternative, not
        assert "(none found)" in out            # a stage with no modules is not silent


class TestMain:
    def test_no_command_prints_help_rather_than_failing(self, capsys):
        with patch.object(pipeline.sys, "argv", ["pipeline.py"]):
            pipeline.main()
        assert "usage:" in capsys.readouterr().out

    def test_a_subcommand_is_dispatched(self, registry):
        with patch.object(pipeline.sys, "argv", ["pipeline.py", "ingest"]):
            with patch.object(pipeline, "cmd_ingest") as cmd:
                pipeline.main()
        cmd.assert_called_once()
