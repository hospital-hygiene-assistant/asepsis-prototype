"""
Unit tests for all deterministic functions in pageindex.py.
No Ollama calls — fast, fully self-contained.

Run: pytest tests/test_pageindex_units.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pageindex.nodes import (
    PageNode,
    _build_tree,
    _clean_node_id,
    _extract_leaf_content,
    _find_nodes_by_ids,
    _flatten_toc,
    _generate_summary,
    _parse_headings,
    _populate_content,
    _populate_summaries,
    _promote_preambles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_DOC = """\
# Alpha
Alpha content here.

## Beta
Beta content here.

## Gamma
Gamma content here.
"""

DEEP_DOC = """\
# Root
## Level Two
### Level Three
#### Level Four
Deep leaf content.
"""

PREAMBLE_DOC = """\
# Section With Preamble
This is preamble text before the children.

## Child One
Child one content.

## Child Two
Child two content.
"""

UNBALANCED_DOC = """\
# Top
Intro text.

## Shallow Leaf
Short content.

## Deep Branch
### Deep Child
Very deep content here.
"""

DUPLICATE_HEADINGS_DOC = """\
# Overview
First overview.

## Details
First details.

## Details
Second details.
"""

LEVEL_JUMP_DOC = """\
# Start
Start content.

### Jump
Jumped to level 3.
"""

BLANK_PREAMBLE_DOC = """\
# Section

## Child
Child content.
"""


def _lines(doc: str) -> list[str]:
    return doc.split("\n")


# ---------------------------------------------------------------------------
# _parse_headings
# ---------------------------------------------------------------------------

class TestParseHeadings:

    def test_basic_levels_and_titles(self):
        headings = _parse_headings(SIMPLE_DOC)
        assert len(headings) == 3
        assert headings[0][1] == 1 and headings[0][2] == "Alpha"
        assert headings[1][1] == 2 and headings[1][2] == "Beta"
        assert headings[2][1] == 2 and headings[2][2] == "Gamma"

    def test_node_id_slugified(self):
        headings = _parse_headings("# Hello World\n## Foo Bar\n")
        assert headings[0][3] == "hello-world"
        assert headings[1][3] == "foo-bar"

    def test_special_chars_slugified(self):
        headings = _parse_headings("## Severity Assessment (CURB-65)\n")
        assert headings[0][3] == "severity-assessment-curb-65"

    def test_duplicate_headings_deduplicated(self):
        headings = _parse_headings(DUPLICATE_HEADINGS_DOC)
        ids = [h[3] for h in headings]
        assert ids[1] == "details"
        assert ids[2] == "details-2"

    def test_no_headings_returns_empty(self):
        assert _parse_headings("Just plain text.\nNo headings here.\n") == []

    def test_level_jump_handled(self):
        headings = _parse_headings(LEVEL_JUMP_DOC)
        assert headings[0][1] == 1
        assert headings[1][1] == 3  # level jump preserved as-is

    def test_line_idx_correct(self):
        headings = _parse_headings(SIMPLE_DOC)
        lines = _lines(SIMPLE_DOC)
        for line_idx, level, title, node_id in headings:
            assert lines[line_idx].startswith("#" * level)
            assert title in lines[line_idx]


# ---------------------------------------------------------------------------
# _build_tree
# ---------------------------------------------------------------------------

class TestBuildTree:

    def test_flat_doc_all_roots(self):
        doc = "# A\nContent A.\n# B\nContent B.\n"
        headings = _parse_headings(doc)
        nodes = _build_tree(headings)
        assert len(nodes) == 2
        assert all(n.is_leaf for n in nodes)

    def test_nested_children(self):
        headings = _parse_headings(SIMPLE_DOC)
        nodes = _build_tree(headings)
        assert len(nodes) == 1
        assert nodes[0].title == "Alpha"
        assert len(nodes[0].children) == 2
        assert nodes[0].children[0].title == "Beta"
        assert nodes[0].children[1].title == "Gamma"

    def test_deep_nesting(self):
        headings = _parse_headings(DEEP_DOC)
        nodes = _build_tree(headings)
        assert len(nodes) == 1
        lv2 = nodes[0].children[0]
        lv3 = lv2.children[0]
        lv4 = lv3.children[0]
        assert lv4.heading_level == 4
        assert lv4.is_leaf

    def test_unbalanced_tree(self):
        headings = _parse_headings(UNBALANCED_DOC)
        nodes = _build_tree(headings)
        root = nodes[0]
        # Shallow Leaf and Deep Branch are siblings
        assert len(root.children) == 2
        assert root.children[0].title == "Shallow Leaf"
        assert root.children[1].title == "Deep Branch"
        assert root.children[1].children[0].title == "Deep Child"

    def test_single_heading(self):
        nodes = _build_tree(_parse_headings("# Solo\nContent.\n"))
        assert len(nodes) == 1
        assert nodes[0].is_leaf

    def test_level_jump_becomes_child(self):
        headings = _parse_headings(LEVEL_JUMP_DOC)
        nodes = _build_tree(headings)
        # level-3 heading is deeper than level-1 → becomes a child
        assert len(nodes) == 1
        assert len(nodes[0].children) == 1
        assert nodes[0].children[0].heading_level == 3


# ---------------------------------------------------------------------------
# _extract_preamble and _promote_preambles
# ---------------------------------------------------------------------------

class TestPreamble:

    def test_preamble_detected_and_extracted(self):
        lines = _lines(PREAMBLE_DOC)
        headings = _parse_headings(PREAMBLE_DOC)
        nodes = _build_tree(headings)
        # Before promotion: root has 2 children
        assert len(nodes[0].children) == 2
        _promote_preambles(nodes, lines)
        # After promotion: synthetic leaf inserted as first child
        assert nodes[0].children[0].synthetic is True
        assert "Overview" in nodes[0].children[0].title
        assert "preamble text" in nodes[0].children[0].content

    def test_no_preamble_no_synthetic_leaf(self):
        doc = "# Section\n## Child\nChild content.\n"
        lines = _lines(doc)
        nodes = _build_tree(_parse_headings(doc))
        _promote_preambles(nodes, lines)
        # No non-blank text between # Section and ## Child
        assert not any(c.synthetic for c in nodes[0].children)

    def test_pure_leaf_unchanged(self):
        doc = "# Leaf\nLeaf content.\n"
        lines = _lines(doc)
        nodes = _build_tree(_parse_headings(doc))
        _promote_preambles(nodes, lines)
        assert nodes[0].is_leaf
        assert nodes[0].synthetic is False

    def test_blank_only_preamble_not_promoted(self):
        lines = _lines(BLANK_PREAMBLE_DOC)
        nodes = _build_tree(_parse_headings(BLANK_PREAMBLE_DOC))
        _promote_preambles(nodes, lines)
        assert not any(c.synthetic for c in nodes[0].children)

    def test_synthetic_node_ids_are_unique(self):
        lines = _lines(PREAMBLE_DOC)
        nodes = _build_tree(_parse_headings(PREAMBLE_DOC))
        _promote_preambles(nodes, lines)
        synthetic = nodes[0].children[0]
        assert synthetic.node_id.endswith("-overview")
        assert synthetic.node_id != nodes[0].node_id


# ---------------------------------------------------------------------------
# _extract_leaf_content
# ---------------------------------------------------------------------------

class TestExtractLeafContent:

    def test_content_ends_at_next_heading(self):
        lines = _lines(SIMPLE_DOC)
        headings = _parse_headings(SIMPLE_DOC)
        # Beta is at headings[1]: line_idx, level=2
        line_idx, level, _, _ = headings[1]
        content = _extract_leaf_content(line_idx, level, lines)
        assert "Beta content" in content
        assert "Gamma" not in content

    def test_last_leaf_runs_to_eof(self):
        lines = _lines(SIMPLE_DOC)
        headings = _parse_headings(SIMPLE_DOC)
        line_idx, level, _, _ = headings[2]
        content = _extract_leaf_content(line_idx, level, lines)
        assert "Gamma content" in content

    def test_deep_leaf_bounded_by_higher_level(self):
        lines = _lines(DEEP_DOC)
        headings = _parse_headings(DEEP_DOC)
        # Level 4 heading is the last one
        line_idx, level, _, _ = headings[-1]
        content = _extract_leaf_content(line_idx, level, lines)
        assert "Deep leaf content" in content

    def test_content_does_not_include_heading_line_itself(self):
        lines = _lines(SIMPLE_DOC)
        headings = _parse_headings(SIMPLE_DOC)
        line_idx, level, title, _ = headings[1]
        content = _extract_leaf_content(line_idx, level, lines)
        assert not content.startswith("##")


# ---------------------------------------------------------------------------
# _generate_summary
# ---------------------------------------------------------------------------

class TestGenerateSummary:

    def _leaf(self, content: str) -> PageNode:
        return PageNode(node_id="test", title="Test", heading_level=1, line_idx=0,
                        summary="", content=content)

    def _internal(self, child_titles: list[str]) -> PageNode:
        children = [
            PageNode(node_id=f"c{i}", title=t, heading_level=2, line_idx=i, summary="", content="x")
            for i, t in enumerate(child_titles)
        ]
        return PageNode(node_id="parent", title="Parent", heading_level=1, line_idx=0,
                        summary="", children=children)

    def test_leaf_short_content_no_truncation(self):
        node = self._leaf("Short content.")
        assert _generate_summary(node) == "Short content."

    def test_leaf_long_content_truncated_to_25_words(self):
        words = ["word"] * 30
        node = self._leaf(" ".join(words))
        summary = _generate_summary(node)
        assert len(summary.split()) <= 26  # 25 words + "..."
        assert summary.endswith("...")

    def test_leaf_exactly_25_words_no_ellipsis(self):
        words = ["word"] * 25
        node = self._leaf(" ".join(words))
        summary = _generate_summary(node)
        assert not summary.endswith("...")

    def test_internal_covers_children(self):
        node = self._internal(["Child A", "Child B", "Child C"])
        summary = _generate_summary(node)
        assert "Child A" in summary
        assert "Child B" in summary
        assert summary.startswith("Covers:")

    def test_internal_many_children_truncated(self):
        node = self._internal([f"Child {i}" for i in range(6)])
        summary = _generate_summary(node)
        assert "more" in summary


# ---------------------------------------------------------------------------
# _flatten_toc
# ---------------------------------------------------------------------------

class TestFlattenToc:

    def _build(self, doc: str) -> list[PageNode]:
        lines = _lines(doc)
        nodes = _build_tree(_parse_headings(doc))
        _promote_preambles(nodes, lines)
        _populate_content(nodes, lines)
        _populate_summaries(nodes)
        return nodes

    def test_leaf_marked_as_leaf(self):
        nodes = self._build("# A\nContent.\n")
        toc = _flatten_toc(nodes)
        assert "LEAF" in toc

    def test_internal_marked_as_section(self):
        nodes = self._build(SIMPLE_DOC)
        toc = _flatten_toc(nodes)
        assert "SECTION" in toc

    def test_id_prefix_present(self):
        nodes = self._build("# A\nContent.\n")
        toc = _flatten_toc(nodes)
        assert "id=" in toc

    def test_child_indented_more_than_parent(self):
        nodes = self._build(SIMPLE_DOC)
        toc = _flatten_toc(nodes)
        lines = [l for l in toc.split("\n") if l.strip()]
        parent_indent = len(lines[0]) - len(lines[0].lstrip())
        child_indent = len(lines[1]) - len(lines[1].lstrip())
        assert child_indent > parent_indent

    def test_all_nodes_appear(self):
        nodes = self._build(SIMPLE_DOC)
        toc = _flatten_toc(nodes)
        assert "alpha" in toc
        assert "beta" in toc or "Beta" in toc
        assert "gamma" in toc or "Gamma" in toc


# ---------------------------------------------------------------------------
# _clean_node_id
# ---------------------------------------------------------------------------

class TestCleanNodeId:

    def test_already_clean(self):
        assert _clean_node_id("node-abc") == "node-abc"

    def test_strips_leaf_prefix(self):
        assert _clean_node_id("LEAF node-abc") == "node-abc"

    def test_strips_section_prefix(self):
        assert _clean_node_id("SECTION node-abc") == "node-abc"

    def test_strips_id_equals(self):
        assert _clean_node_id("id=node-abc") == "node-abc"

    def test_strips_path_prefix(self):
        assert _clean_node_id("parent-section/id=node-abc") == "node-abc"

    def test_strips_backslash_and_colon(self):
        assert _clean_node_id("SECTION\\: node-abc") == "node-abc"

    def test_strips_bracket_prefix(self):
        assert _clean_node_id("[LEAF] node-abc") == "node-abc"

    def test_extracts_id_from_toc_line(self):
        assert _clean_node_id("LEAF | id=node-abc | Title | Summary") == "node-abc"


# ---------------------------------------------------------------------------
# _find_nodes_by_ids
# ---------------------------------------------------------------------------

class TestFindNodesByIds:

    def _tree(self) -> list[PageNode]:
        """
        root (internal)
          ├── leaf-a (leaf)
          └── branch (internal)
                ├── leaf-b (leaf)
                └── leaf-c (leaf)
        """
        leaf_a = PageNode("leaf-a", "Leaf A", 2, 1, "sum", content="Content A")
        leaf_b = PageNode("leaf-b", "Leaf B", 3, 3, "sum", content="Content B")
        leaf_c = PageNode("leaf-c", "Leaf C", 3, 5, "sum", content="Content C")
        branch = PageNode("branch", "Branch", 2, 2, "sum", children=[leaf_b, leaf_c])
        root = PageNode("root", "Root", 1, 0, "sum", children=[leaf_a, branch])
        return [root]

    def test_select_leaf_returns_leaf(self):
        result = _find_nodes_by_ids(self._tree(), {"leaf-a"})
        assert len(result) == 1
        assert result[0].node_id == "leaf-a"

    def test_select_internal_returns_all_descendants(self):
        result = _find_nodes_by_ids(self._tree(), {"branch"})
        ids = {n.node_id for n in result}
        assert ids == {"leaf-b", "leaf-c"}

    def test_select_root_returns_all_leaves(self):
        result = _find_nodes_by_ids(self._tree(), {"root"})
        ids = {n.node_id for n in result}
        assert ids == {"leaf-a", "leaf-b", "leaf-c"}

    def test_nonexistent_id_ignored(self):
        result = _find_nodes_by_ids(self._tree(), {"does-not-exist"})
        assert result == []

    def test_mix_of_leaf_and_internal_no_duplicates(self):
        # selecting both "branch" and "leaf-b" — leaf-b should appear once
        result = _find_nodes_by_ids(self._tree(), {"branch", "leaf-a"})
        ids = [n.node_id for n in result]
        assert len(ids) == len(set(ids))  # no duplicates
        assert "leaf-a" in ids
        assert "leaf-b" in ids
        assert "leaf-c" in ids
