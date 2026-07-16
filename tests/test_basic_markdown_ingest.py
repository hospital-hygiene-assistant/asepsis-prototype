"""The Markdown ingest adapter cannot inherit stale PDF provenance."""

from modules.ingest import basic_markdown
from pageindex.library import (
    SourceCandidate,
    read_source_candidates,
    write_source_candidates,
)


def test_markdown_ingest_removes_a_stale_pdf_candidate(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    kb = tmp_path / "kb"
    docs.mkdir()
    kb.mkdir()
    (docs / "guide.md").write_text("# New guide\n\nText.", encoding="utf-8")
    old_pdf = tmp_path / "old.pdf"
    old_pdf.write_bytes(b"%PDF-1.7\nold\n")
    manifest = kb / ".sources.json"
    write_source_candidates(
        manifest,
        {"guide": SourceCandidate(old_pdf, ocr_scale=2.0)},
    )
    monkeypatch.setattr(basic_markdown, "DOCS_DIR", docs)
    monkeypatch.setattr(basic_markdown, "KB_DIR", kb)
    monkeypatch.setattr(basic_markdown, "SOURCES_MANIFEST", manifest)

    basic_markdown.run()

    assert "guide" not in read_source_candidates(manifest)
