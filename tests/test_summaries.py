"""
Phase 1 — LLM summaries at index time.

Two things are being pinned down here:

1. Quality of input. Section summaries must be built from their children's
   SUMMARIES, not from raw content and not from a recursive dump of descendant
   headings. That is what makes Phase 2's batched pruning possible.
2. Cost. Summaries are computed once and persisted. Re-indexing an unchanged
   document must cost zero LLM calls; editing one leaf must re-summarise that
   leaf and its ancestor chain, and nothing else.
"""
import json

import pytest

import config as app_config
import pageindex

pytestmark = pytest.mark.unit


def _summariser(recorder, text=None):
    """Respond to summary prompts only; anything else keeps the default.

    With no fixed `text`, the response is derived from the prompt, so a
    changed passage produces a changed summary — the way a real summariser
    behaves, and what makes the ancestor-invalidation chain observable.
    """
    def handler(prompt, model):
        if "Write the summary now:" not in prompt:
            return None
        if text is not None:
            return text
        import hashlib
        return f"Summary-{hashlib.sha256(prompt.encode()).hexdigest()[:8]}"
    recorder.handler = handler
    return recorder


def _build(corpus, recorder, **kw):
    _summariser(recorder)
    return pageindex.build_index(corpus["doc"], **kw)


class TestSummaryContent:
    def test_leaf_summary_prompt_carries_the_content(self, tmp_corpus, fake_ollama):
        _build(tmp_corpus, fake_ollama)
        leaf_prompts = fake_ollama.prompts_containing("--- PASSAGE ---")
        assert leaf_prompts, "leaves must be summarised from their content"
        assert any("amoxicillin 500mg" in p for p in leaf_prompts)

    def test_section_summary_is_built_from_child_summaries(self, tmp_corpus, fake_ollama):
        _summariser(fake_ollama, "CHILD-SUMMARY-MARKER")
        pageindex.build_index(tmp_corpus["doc"])

        section_prompts = fake_ollama.prompts_containing("SUMMARIES OF THIS SECTION'S PARTS")
        assert section_prompts, "sections must be summarised too"
        for p in section_prompts:
            assert "CHILD-SUMMARY-MARKER" in p, "section prompt must contain child summaries"
            assert "amoxicillin 500mg" not in p, (
                "section prompt must NOT contain raw child content — summaries "
                "are the whole point of the bottom-up pass")

    def test_bottom_up_order(self, tmp_corpus, fake_ollama):
        """Children must be summarised before their parent, or the parent's
        prompt would contain empty summaries."""
        _build(tmp_corpus, fake_ollama)
        kinds = ["leaf" if "--- PASSAGE ---" in p else "section"
                 for p in fake_ollama.prompts]
        assert kinds[0] == "leaf"
        assert kinds[-1] == "section", "the root is summarised last"

    def test_synthetic_overview_leaf_is_summarised_and_reaches_its_parent(
            self, tmp_corpus, fake_ollama):
        """The section's own prose used to be dropped from its parent summary
        entirely — it is content, and it must be represented."""
        _summariser(fake_ollama)
        pageindex.build_index(tmp_corpus["doc"])

        nodes = pageindex.load_index_nodes(tmp_corpus["doc"])
        overviews = [n for n in pageindex._collect_leaves(nodes) if n.synthetic]
        assert overviews, "fixture should produce at least one overview leaf"
        assert all(o.summary for o in overviews)

        section_prompts = fake_ollama.prompts_containing("SUMMARIES OF THIS SECTION'S PARTS")
        assert any("(section prose)" in p for p in section_prompts)

    def test_summary_is_cleaned_of_fences_and_whitespace(self, tmp_corpus, fake_ollama):
        _summariser(fake_ollama, "```\n  Multi\n  line   summary.  \n```")
        pageindex.build_index(tmp_corpus["doc"])
        nodes = pageindex.load_index_nodes(tmp_corpus["doc"])
        assert nodes[0].summary == "Multi line summary."


class TestPersistence:
    def test_summaries_are_written_with_provenance(self, tmp_corpus, fake_ollama):
        _build(tmp_corpus, fake_ollama)
        data = json.loads((tmp_corpus["index"] / "guideline.json").read_text())

        assert data["formatVersion"] == app_config.INDEX_FORMAT_VERSION
        root = data["nodes"][0]
        assert root["summary"]
        assert root["contentHash"]
        assert root["summarySource"] == "llm"
        assert root["summaryModel"] == pageindex.MODEL
        assert root["summaryPromptVersion"] == pageindex._summary_prompt_version()

    def test_index_is_not_a_bare_array_anymore(self, tmp_corpus, fake_ollama):
        _build(tmp_corpus, fake_ollama)
        data = json.loads((tmp_corpus["index"] / "guideline.json").read_text())
        assert isinstance(data, dict) and "nodes" in data

    def test_v1_bare_array_still_reads(self, tmp_path):
        """Old indexes must be readable (so they can be detected as stale)
        rather than crashing the loader."""
        p = tmp_path / "old.json"
        p.write_text(json.dumps([{
            "nodeId": "a", "title": "A", "headingLevel": 1, "lineIdx": 0,
            "summary": "s", "isLeaf": True, "children": [],
        }]))
        data = pageindex.read_index_file(p)
        assert data["formatVersion"] == 1
        assert len(data["nodes"]) == 1
        assert pageindex.index_is_stale(p) is True


class TestIncrementalRebuild:
    def test_unchanged_document_costs_zero_llm_calls(self, tmp_corpus, fake_ollama):
        _build(tmp_corpus, fake_ollama)
        assert fake_ollama.count > 0
        fake_ollama.reset()

        report = pageindex.build_index(tmp_corpus["doc"])
        fake_ollama.assert_count(0, "LLM calls on an unchanged re-index")
        assert report["summaries_generated"] == 0
        assert report["summaries_reused"] > 0

    def test_editing_one_leaf_resummarises_only_its_ancestor_chain(
            self, tmp_corpus, fake_ollama):
        _build(tmp_corpus, fake_ollama)
        fake_ollama.reset()

        md = tmp_corpus["kb"] / "guideline.md"
        md.write_text(md.read_text().replace(
            "CT is preferred over ultrasound for this indication.",
            "MRI is now preferred over CT for this indication."), encoding="utf-8")

        report = pageindex.build_index(tmp_corpus["doc"])

        # Imaging leaf + Diagnosis section + Guideline root = 3.
        assert report["summaries_generated"] == 3, (
            f"expected the edited leaf and its 2 ancestors, got "
            f"{report['summaries_generated']}")
        assert report["summaries_reused"] > 0

    def test_ancestors_are_spared_when_the_child_summary_is_unchanged(
            self, tmp_corpus, fake_ollama):
        """Invalidation keys on the child's SUMMARY, not on its raw content.
        A wording change that the summariser renders identically stops there
        instead of dirtying the whole chain to the root."""
        _summariser(fake_ollama, "A fixed summary.")
        pageindex.build_index(tmp_corpus["doc"])
        fake_ollama.reset()

        md = tmp_corpus["kb"] / "guideline.md"
        md.write_text(md.read_text().replace(
            "CT is preferred over ultrasound for this indication.",
            "CT is preferred over ultrasound in this indication."), encoding="utf-8")

        report = pageindex.build_index(tmp_corpus["doc"])
        assert report["summaries_generated"] == 1, "only the edited leaf"

    def test_model_change_invalidates_every_summary(self, tmp_corpus, fake_ollama, monkeypatch):
        _build(tmp_corpus, fake_ollama)
        before = fake_ollama.count
        fake_ollama.reset()

        monkeypatch.setattr(pageindex, "MODEL", "some-other-model")
        report = pageindex.build_index(tmp_corpus["doc"])
        assert report["summaries_reused"] == 0
        assert fake_ollama.count == before

    def test_prompt_version_bump_invalidates_every_summary(
            self, tmp_corpus, fake_ollama, monkeypatch):
        _build(tmp_corpus, fake_ollama)
        fake_ollama.reset()

        bumped = dict(app_config.PROMPT_VERSIONS)
        bumped["leaf_summary"] += 1
        monkeypatch.setattr(app_config, "PROMPT_VERSIONS", bumped)

        report = pageindex.build_index(tmp_corpus["doc"])
        assert report["summaries_reused"] == 0

    def test_heuristic_summaries_are_never_reused(self, tmp_corpus, fake_ollama, monkeypatch):
        """A degraded build must not poison later builds with its fallbacks."""
        real_chat = pageindex._chat
        down = {"value": True}

        def flaky(*a, **kw):
            if down["value"]:
                raise ConnectionError("down")
            return real_chat(*a, **kw)

        monkeypatch.setattr(pageindex, "_chat", flaky)
        first = pageindex.build_index(tmp_corpus["doc"])
        assert first["summaries_heuristic"] > 0

        down["value"] = False
        _summariser(fake_ollama)
        report = pageindex.build_index(tmp_corpus["doc"])
        assert report["summaries_reused"] == 0, (
            "heuristic fallbacks must not be treated as valid cached summaries")
        assert report["summaries_generated"] > 0


class TestGracefulDegradation:
    def test_falls_back_to_heuristic_when_ollama_is_down(self, tmp_corpus, monkeypatch):
        monkeypatch.setattr(pageindex, "_chat",
                            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("refused")))
        report = pageindex.build_index(tmp_corpus["doc"])

        assert report["summaries_heuristic"] > 0
        assert report["errors"], "degradation must be reported, not silent"
        nodes = pageindex.load_index_nodes(tmp_corpus["doc"])
        assert nodes[0].summary, "index is still usable"
        assert nodes[0].summary_source == "heuristic"

    def test_empty_leaf_uses_heuristic_without_calling_the_model(
            self, tmp_corpus, fake_ollama):
        node = pageindex.PageNode(node_id="e", title="Empty", heading_level=1,
                                  line_idx=0, summary="", content="   ")
        summary, source = pageindex._summarise_node(node, "Empty")
        assert source == "heuristic"
        fake_ollama.assert_count(0)

    def test_use_llm_false_makes_no_calls(self, tmp_corpus, fake_ollama):
        pageindex.build_index(tmp_corpus["doc"], use_llm_summaries=False)
        fake_ollama.assert_count(0)


class TestHeuristicFallbackShape:
    def test_overview_leaves_are_excluded_from_the_count(self):
        """The original sliced the unfiltered child list for 'and N more'
        while filtering synthetic children out of the titles, so the count
        disagreed with the list."""
        children = [
            pageindex.PageNode(node_id="ov", title="Parent — Overview", heading_level=2,
                               line_idx=0, summary="", content="x", synthetic=True),
        ] + [
            pageindex.PageNode(node_id=f"c{i}", title=f"Child {i}", heading_level=2,
                               line_idx=i, summary="", content="x")
            for i in range(6)
        ]
        parent = pageindex.PageNode(node_id="p", title="Parent", heading_level=1,
                                    line_idx=0, summary="", children=children)
        summary = pageindex._generate_summary(parent)
        assert "Overview" not in summary
        assert "and 2 more" in summary, f"6 real children, 4 listed → 2 more; got {summary!r}"
