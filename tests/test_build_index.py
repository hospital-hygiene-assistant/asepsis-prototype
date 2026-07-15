"""Building a document's tree from its markdown.

Wholly deterministic — no model is consulted — which makes it the one part of
the pipeline that can be pinned exactly. It was covered at 17%: an index built
wrong is not a visible failure, it is an answer citing the wrong passage, or a
document that silently contributes nothing.
"""

import json

import pytest

from pageindex import build
from pageindex.build import build_index
from pageindex.nodes import _node_from_dict

DOC = """\
# Hygiene Guideline

Intro text before any subsection.

## Hand Hygiene

```pin
id: hand-hygiene
kind: section
doc: hygiene
page: 3
bbox: [50.0, 80.0, 400.0, 110.0]
regions: [[3, 50.0, 200.0, 500.0, 460.0]]
scale: 2.0
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
    kb, index = tmp_path / "kb", tmp_path / "index"
    kb.mkdir()
    monkeypatch.setattr(build, "KB_DIR", kb)
    monkeypatch.setattr(build, "INDEX_DIR", index)
    (kb / "hygiene.md").write_text(DOC, encoding="utf-8")
    build_index("hygiene")
    return json.loads((index / "hygiene.json").read_text(encoding="utf-8"))


def find(nodes: list[dict], node_id: str) -> dict | None:
    for n in nodes:
        if n["nodeId"] == node_id:
            return n
        hit = find(n.get("children") or [], node_id)
        if hit:
            return hit
    return None


class TestBuildIndex:
    def test_a_missing_document_is_an_error_not_an_empty_index(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build, "KB_DIR", tmp_path)
        monkeypatch.setattr(build, "INDEX_DIR", tmp_path / "index")
        with pytest.raises(FileNotFoundError, match="Document not found"):
            build_index("never-ingested")

    def test_the_index_directory_is_created_on_demand(self, built, tmp_path):
        assert (tmp_path / "index" / "hygiene.json").is_file()

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

        The bbox is written by _massage._fmt_bbox as a JSON array, and
        api/pdf._normalized_bbox requires a list — a bbox left as a bare string
        would silently yield no highlight, so the parse to real numbers matters.
        """
        pin = find(built, "hand-hygiene")["pin"]
        assert pin is not None
        assert pin["page"] == 3
        assert pin["bbox"] == [50.0, 80.0, 400.0, 110.0]
        assert pin["regions"] == [[3, 50.0, 200.0, 500.0, 460.0]]

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
