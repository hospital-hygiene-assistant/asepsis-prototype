"""Building a document's tree from its markdown.

Wholly deterministic — no model is consulted — which makes it the one part of
the pipeline that can be pinned exactly. It was covered at 17%: an index built
wrong is not a visible failure, it is an answer citing the wrong passage, or a
document that silently contributes nothing.
"""

import json

import pytest

from pageindex import build
from pageindex.build import build_generation
from pageindex.nodes import _node_from_dict
from pageindex.library import (
    ExpectedLibraryStore,
    SourceCandidate,
    write_source_candidates,
)

DOC = """\
# Hygiene Guideline

Intro text before any subsection.

## Hand Hygiene

```pin
{"version":2,"document":"hygiene","nodeId":"hand-hygiene","spans":[{"page":3,"start":0,"end":54,"box":[50.0,200.0,500.0,460.0]}],"scale":2.0}
```

Disinfect hands before and after every patient contact.

### Technique

Rub for thirty seconds.

## Isolation

Single room for confirmed MRSA.
"""


@pytest.fixture
def built(tmp_path, monkeypatch):
    """Build DOC through the real entrypoint and hand back the parsed index."""
    kb, library = tmp_path / "kb", tmp_path / "library"
    kb.mkdir()
    monkeypatch.setattr(build, "KB_DIR", kb)
    monkeypatch.setattr(build, "LIBRARY_DIR", library)
    monkeypatch.setattr(build, "SOURCES_MANIFEST", kb / ".sources.json")
    (kb / "hygiene.md").write_text(DOC, encoding="utf-8")
    source_pdf = tmp_path / "hygiene.pdf"
    source_pdf.write_bytes(b"%PDF-1.7\nsource\n")
    write_source_candidates(
        kb / ".sources.json",
        {"hygiene": SourceCandidate(source_pdf, ocr_scale=2.0)},
    )
    snapshot = build_generation(["hygiene"])
    return json.loads(
        snapshot.document("hygiene").index_path.read_text(encoding="utf-8")
    )


def find(nodes: list[dict], node_id: str) -> dict | None:
    for n in nodes:
        if n["nodeId"] == node_id:
            return n
        hit = find(n.get("children") or [], node_id)
        if hit:
            return hit
    return None


class TestBuildGeneration:
    def test_a_missing_document_is_an_error_not_an_empty_index(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build, "KB_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="Document not found"):
            build_generation(["never-ingested"])

    def test_headings_become_the_hierarchy(self, built):
        root = built[0]
        assert root["nodeId"] == "hygiene-guideline"
        assert [c["nodeId"] for c in root["children"] if not c["synthetic"]] == [
            "hand-hygiene", "isolation",
        ]
        assert find(built, "technique") is not None, "### should nest under its ##"

    def test_a_leaf_carries_its_own_text(self, built):
        assert "Rub for thirty seconds." in find(built, "technique")["content"]

    def test_text_before_the_first_subsection_is_not_lost(self, built):
        """The intro belongs to no heading of its own, so it is promoted to a
        synthetic leaf — otherwise it could never be retrieved or cited."""
        synthetic = [c for c in built[0]["children"] if c["synthetic"]]
        assert synthetic, "the preamble should survive as a leaf"
        assert "Intro text before any subsection." in synthetic[0]["content"]

    def test_a_pin_is_lifted_onto_its_node(self, built):
        """The pin is the whole provenance chain's first link: without it a
        citation cannot point back into the source PDF at all.

        The source span is written by the real v2 serializer. Reloading it into
        typed coordinates matters because a malformed span must never become a
        guessed highlight.
        """
        pin = find(built, "hand-hygiene")["pin"]
        assert pin is not None
        assert pin["version"] == 2
        assert pin["nodeId"] == "hand-hygiene"
        assert pin["spans"] == [{
            "page": 3,
            "start": 0,
            "end": 54,
            "box": [50.0, 200.0, 500.0, 460.0],
        }]

    def test_the_pin_block_is_not_left_in_the_readable_content(self, built):
        # It is machine metadata; leaving it in would put fence noise in front of
        # the model and into any excerpt shown to a clinician.
        content = find(built, "hand-hygiene")["content"] or ""
        assert "```pin" not in content and "bbox:" not in content

    def test_a_section_summarises_its_children_and_a_leaf_its_text(self, built):
        assert find(built, "hygiene-guideline")["summary"].startswith("Covers:")
        assert "Rub for thirty seconds" in find(built, "technique")["summary"]

    def test_the_written_index_reloads_into_real_nodes(self, built):
        """The file is the contract between build and retrieve; if it cannot be
        read back by the engine's own loader, nothing downstream works."""
        nodes = [_node_from_dict(d) for d in built]
        assert nodes[0].node_id == "hygiene-guideline"
        assert nodes[0].children, "the hierarchy must survive the round trip"


def test_build_generation_promotes_the_corpus_only_after_every_document_builds(
    tmp_path, monkeypatch
):
    kb, index = tmp_path / "kb", tmp_path / "index"
    kb.mkdir()
    monkeypatch.setattr(build, "KB_DIR", kb)
    monkeypatch.setattr(build, "LIBRARY_DIR", index)
    monkeypatch.setattr(build, "SOURCES_MANIFEST", kb / ".sources.json")
    (kb / "a.md").write_text("# A\n\nFirst passage.", encoding="utf-8")
    (kb / "b.md").write_text("# B\n\nSecond passage.", encoding="utf-8")

    snapshot = build_generation(["b", "a"])

    pointer = json.loads((index / "current.json").read_text(encoding="utf-8"))
    assert pointer["generation_id"] == snapshot.generation_id
    assert [document.document_id for document in snapshot.documents] == ["a", "b"]
    assert snapshot.document("a").canonical_markdown == "# A\n\nFirst passage."
    assert (
        ExpectedLibraryStore(index).open_current().generation_id
        == snapshot.generation_id
    )


def test_build_generation_binds_ingest_source_candidate_metadata(
    tmp_path, monkeypatch
):
    kb, index = tmp_path / "kb", tmp_path / "index"
    kb.mkdir()
    source_pdf = tmp_path / "incoming.pdf"
    source_pdf.write_bytes(b"%PDF-1.7\nsource bytes\n")
    markdown = "# Guide\n\nSource text."
    (kb / "guide.md").write_text(markdown, encoding="utf-8")
    manifest = kb / ".sources.json"
    write_source_candidates(
        manifest,
        {"guide": SourceCandidate(source_pdf, ocr_scale=2.0)},
    )
    monkeypatch.setattr(build, "KB_DIR", kb)
    monkeypatch.setattr(build, "LIBRARY_DIR", index)
    monkeypatch.setattr(build, "SOURCES_MANIFEST", manifest)

    snapshot = build_generation(["guide"])

    document = snapshot.document("guide")
    assert document.source is not None
    assert document.source.pdf_path != source_pdf
    assert document.source.pdf_path.read_bytes() == source_pdf.read_bytes()
