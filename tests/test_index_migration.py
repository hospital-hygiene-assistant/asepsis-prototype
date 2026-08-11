"""
Phase 1.5 — forced re-index migration.

v1 indexes are bare node arrays with no summaries and no content hashes. They
cannot be upgraded in place: the summaries retrieval now depends on simply do
not exist in them. So a stale corpus is detected and blocks retrieval until it
is rebuilt — once.
"""
import json

import pytest

import config as app_config
import pageindex

pytestmark = pytest.mark.unit


V1_NODE = {
    "nodeId": "alpha", "title": "Alpha", "headingLevel": 1, "lineIdx": 0,
    "summary": "Covers: Beta, Gamma", "isLeaf": False, "children": [
        {"nodeId": "beta", "title": "Beta", "headingLevel": 2, "lineIdx": 2,
         "summary": "Beta content here...", "isLeaf": True, "children": [],
         "content": "Beta content here."},
    ],
}


def _write_v1(index_dir, stem="legacy"):
    p = index_dir / f"{stem}.json"
    p.write_text(json.dumps([V1_NODE]), encoding="utf-8")
    return p


class TestStaleDetection:
    def test_v1_index_is_stale(self, tmp_corpus):
        p = _write_v1(tmp_corpus["index"])
        assert pageindex.index_is_stale(p) is True
        assert "legacy" in pageindex.stale_indexes(tmp_corpus["index"])

    def test_freshly_built_index_is_not_stale(self, tmp_corpus, fake_ollama):
        pageindex.build_index(tmp_corpus["doc"])
        assert pageindex.stale_indexes(tmp_corpus["index"]) == []

    def test_corrupt_index_counts_as_stale(self, tmp_corpus):
        p = tmp_corpus["index"] / "broken.json"
        p.write_text("{not json", encoding="utf-8")
        assert pageindex.index_is_stale(p) is True

    def test_future_version_is_not_treated_as_stale(self, tmp_corpus):
        p = tmp_corpus["index"] / "future.json"
        p.write_text(json.dumps({
            "formatVersion": app_config.INDEX_FORMAT_VERSION + 5, "nodes": []}),
            encoding="utf-8")
        assert pageindex.index_is_stale(p) is False

    def test_empty_index_dir_is_not_stale(self, tmp_corpus):
        assert pageindex.stale_indexes(tmp_corpus["index"]) == []


class TestRebuild:
    def test_rebuild_clears_staleness_and_happens_once(self, tmp_corpus, fake_ollama):
        _write_v1(tmp_corpus["index"], stem="guideline")
        assert pageindex.stale_indexes(tmp_corpus["index"]) == ["guideline"]

        report = pageindex.build_index(tmp_corpus["doc"])
        assert pageindex.stale_indexes(tmp_corpus["index"]) == []
        assert report["summaries_generated"] > 0
        assert report["summaries_reused"] == 0, (
            "v1 nodes carry no content hashes, so nothing may be reused")

        fake_ollama.reset()
        second = pageindex.build_index(tmp_corpus["doc"])
        assert second["summaries_generated"] == 0
        fake_ollama.assert_count(0, "LLM calls on the second start")

    def test_v1_summaries_are_not_silently_kept(self, tmp_corpus, fake_ollama):
        """The old 'Covers: Beta, Gamma' strings must not survive the rebuild —
        carrying them forward would leave pruning running on the same
        near-signal-free text this phase exists to replace."""
        _write_v1(tmp_corpus["index"], stem="guideline")
        pageindex.build_index(tmp_corpus["doc"])
        nodes = pageindex.load_index_nodes(tmp_corpus["doc"])
        assert not nodes[0].summary.startswith("Covers:")
        assert nodes[0].summary_source == "llm"


class TestLoaderCompat:
    def test_load_index_nodes_reads_both_formats(self, tmp_corpus, fake_ollama):
        _write_v1(tmp_corpus["index"], stem="legacy")
        legacy = pageindex.load_index_nodes("legacy")
        assert legacy[0].node_id == "alpha"

        pageindex.build_index(tmp_corpus["doc"])
        current = pageindex.load_index_nodes(tmp_corpus["doc"])
        assert current[0].content_hash, "v2 nodes carry hashes"

    def test_missing_index_raises_filenotfound(self, tmp_corpus):
        with pytest.raises(FileNotFoundError):
            pageindex.load_index_nodes("nope")
