"""
Phase 4 — recursive ingest, document identity, and folder tags.

Ingest used to look only at the top level of one folder, so using the filename
stem as the document id was survivable. Recursion breaks that: the knowledge
base is flat (`knowledge_base/<id>.md`, `index/<id>.json`), so two files named
`report.pdf` in different folders would silently overwrite each other. The id
is therefore derived from the path relative to the ingest root.
"""
import json

import pytest

from modules.ingest._manifest import (
    Manifest, DocEntry, discover_files, make_doc_id, sanitise, tags_for,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def tree(tmp_path):
    """papers/arxiv/attention.pdf, papers/internal/attention.pdf, loose.pdf"""
    root = tmp_path / "corpus"
    for rel in ("papers/arxiv/attention.pdf",
                "papers/internal/attention.pdf",
                "guidelines/2024/sepsis.pdf",
                "loose.pdf"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"%PDF-1.4 " + rel.encode())
    (root / ".ocr_cache").mkdir()
    (root / ".ocr_cache" / "cached.pdf").write_bytes(b"junk")
    return root


class TestDiscovery:
    def test_finds_files_recursively(self, tree):
        found = discover_files(tree, (".pdf",))
        assert len(found) == 4, "sub-folders must be searched"

    def test_skips_hidden_directories(self, tree):
        names = [p.name for p in discover_files(tree, (".pdf",))]
        assert "cached.pdf" not in names, "caches must never be ingested as documents"

    def test_is_deterministic(self, tree):
        assert discover_files(tree, (".pdf",)) == discover_files(tree, (".pdf",))

    def test_suffix_filter(self, tree):
        (tree / "notes.md").write_text("# hi")
        assert [p.name for p in discover_files(tree, (".md",))] == ["notes.md"]


class TestDocId:
    def test_derived_from_relative_path(self, tree):
        doc_id = make_doc_id(tree / "papers/arxiv/attention.pdf", tree)
        assert doc_id == "papers__arxiv__attention"

    def test_same_stem_in_different_folders_does_not_collide(self, tree):
        a = make_doc_id(tree / "papers/arxiv/attention.pdf", tree)
        b = make_doc_id(tree / "papers/internal/attention.pdf", tree)
        assert a != b, "this collision silently destroyed a document before"

    def test_file_at_root_has_a_bare_id(self, tree):
        assert make_doc_id(tree / "loose.pdf", tree) == "loose"

    def test_unsafe_characters_are_sanitised(self, tmp_path):
        p = tmp_path / "wei rd & na:me" / "a b.pdf"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"x")
        doc_id = make_doc_id(p, tmp_path)
        assert "/" not in doc_id and " " not in doc_id and ":" not in doc_id

    def test_sanitisation_collisions_get_a_suffix(self, tmp_path):
        """'a b/x' and 'a-b/x' both sanitise to 'a_b__x' — one must not win."""
        taken = {"a_b__x"}
        assert make_doc_id(tmp_path / "a b" / "x.pdf", tmp_path, taken=taken) != "a_b__x"

    def test_empty_name_falls_back(self):
        assert sanitise("...") == "untitled"


class TestFolderTags:
    def test_every_path_component_becomes_a_tag(self, tree):
        assert tags_for(tree / "papers/arxiv/attention.pdf", tree) == ["papers", "arxiv"]

    def test_nested_folders_give_broad_and_narrow_tags(self, tree):
        tags = tags_for(tree / "guidelines/2024/sepsis.pdf", tree)
        assert "guidelines" in tags and "2024" in tags

    def test_root_level_file_has_no_folder_tags(self, tree):
        assert tags_for(tree / "loose.pdf", tree) == []


class TestManifest:
    def test_register_records_id_tags_and_hash(self, tmp_path, tree):
        m = Manifest(tmp_path / "kb")
        entry = m.register(source_path=tree / "papers/arxiv/attention.pdf",
                           root=tree, ingest_module="betteringest_pdf")
        assert entry.doc_id == "papers__arxiv__attention"
        assert entry.title == "attention"
        assert entry.effective_tags == ["arxiv", "papers"]
        assert len(entry.sha256) == 64

    def test_persists_and_reloads(self, tmp_path, tree):
        kb = tmp_path / "kb"
        Manifest(kb).register(source_path=tree / "loose.pdf", root=tree,
                              ingest_module="x")
        assert (kb / ".manifest.json").exists()
        assert "loose" in Manifest(kb).entries

    def test_manual_tags_survive_a_reingest(self, tmp_path, tree):
        """Folder tags refresh on re-ingest; curation must not be clobbered."""
        kb = tmp_path / "kb"
        m = Manifest(kb)
        src = tree / "papers/arxiv/attention.pdf"
        m.register(source_path=src, root=tree, ingest_module="x")
        m.set_manual_tags("papers__arxiv__attention", ["landmark", "reviewed"])

        m2 = Manifest(kb)
        entry = m2.register(source_path=src, root=tree, ingest_module="x")
        assert set(entry.manual_tags) == {"landmark", "reviewed"}
        assert "arxiv" in entry.effective_tags, "folder tags are still derived"

    def test_reingest_reuses_the_existing_id(self, tmp_path, tree):
        kb = tmp_path / "kb"
        m = Manifest(kb)
        src = tree / "loose.pdf"
        first = m.register(source_path=src, root=tree, ingest_module="x")
        second = m.register(source_path=src, root=tree, ingest_module="x")
        assert first.doc_id == second.doc_id
        assert len(m.entries) == 1

    def test_all_tags_counts_documents(self, tmp_path, tree):
        m = Manifest(tmp_path / "kb")
        for rel in ("papers/arxiv/attention.pdf", "papers/internal/attention.pdf"):
            m.register(source_path=tree / rel, root=tree, ingest_module="x")
        counts = {t["tag"]: t["doc_count"] for t in m.all_tags()}
        assert counts["papers"] == 2
        assert counts["arxiv"] == 1

    def test_filter_docs_is_a_union(self, tmp_path, tree):
        m = Manifest(tmp_path / "kb")
        for rel in ("papers/arxiv/attention.pdf", "guidelines/2024/sepsis.pdf",
                    "loose.pdf"):
            m.register(source_path=tree / rel, root=tree, ingest_module="x")

        assert m.filter_docs(["arxiv"]) == {"papers__arxiv__attention"}
        assert len(m.filter_docs(["arxiv", "guidelines"])) == 2
        assert m.filter_docs([]) == set(m.entries), "no selection means everything"
        assert m.filter_docs(None) == set(m.entries)
        assert m.filter_docs(["nonexistent"]) == set()


class TestLegacyMigration:
    def test_sources_json_is_migrated(self, tmp_path):
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / ".sources.json").write_text(json.dumps({
            "hypertension_guidelines": {
                "pdf": "/abs/hypertension_guidelines.pdf",
                "module": "betteringest_pdf",
                "ocr_scale": 2.0,
                "assets": [],
            }}))

        m = Manifest(kb)
        entry = m.get("hypertension_guidelines")
        assert entry is not None, "an existing library must not be orphaned"
        assert entry.source_path == "/abs/hypertension_guidelines.pdf"
        assert entry.extra["ocr_scale"] == 2.0

    def test_migration_is_lossless_for_ids(self, tmp_path):
        """Pre-recursion, the stem WAS the id — migrated entries must keep it
        so existing index/*.json files still match their metadata."""
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / ".sources.json").write_text(json.dumps({"doc_a": {}, "doc_b": {}}))
        assert set(Manifest(kb).entries) == {"doc_a", "doc_b"}

    def test_manifest_wins_when_both_exist(self, tmp_path):
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / ".sources.json").write_text(json.dumps({"old": {}}))
        (kb / ".manifest.json").write_text(json.dumps({"new": {"title": "New"}}))
        assert set(Manifest(kb).entries) == {"new"}


class TestBasicMarkdownIngest:
    def test_recursive_ingest_writes_namespaced_documents(self, tmp_path, monkeypatch):
        import modules.ingest.basic_markdown as bm

        src = tmp_path / "docs"
        for rel in ("internal/report.md", "arxiv/report.md"):
            p = src / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"# {rel}\n\nbody", encoding="utf-8")

        kb = tmp_path / "kb"
        monkeypatch.setattr(bm, "KB_DIR", kb)
        result = bm.run(source_dir=str(src))

        assert set(result["docs"]) == {"internal__report", "arxiv__report"}
        assert (kb / "internal__report.md").exists()
        assert (kb / "arxiv__report.md").exists(), (
            "the second report.md must not have overwritten the first")

        m = Manifest(kb)
        assert m.tags_of("arxiv__report") == ["arxiv"]
