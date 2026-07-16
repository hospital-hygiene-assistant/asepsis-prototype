import pytest

from pageindex.document_index import DocumentIndex
from pageindex.nodes import PageNode


def test_document_index_owns_navigation_and_debug_projection():
    leaf = PageNode(
        node_id="leaf", title="Leaf", heading_level=2, line_idx=2,
        summary="Guidance", content="Guidance", children=[],
    )
    root = PageNode(
        node_id="root", title="Root", heading_level=1, line_idx=0,
        summary="Leaf", content="", children=[leaf],
    )

    index = DocumentIndex.from_nodes((root,))

    assert index.leaf_count == 1
    assert index.all_nodes == (root, leaf)
    assert index.descendants("root") == (leaf,)
    assert index.descendants("root", include_self=True) == (root, leaf)
    assert index.leaves_under("root") == (leaf,)
    assert index.parent_summary("leaf") == "Leaf"
    assert index.parent_summary("root") == ""
    assert index.node("leaf").content == "Guidance"
    assert index.breadcrumb("leaf") == "Root > Leaf"
    assert index.debug_tree[0]["children"][0]["nodeId"] == "leaf"
    with pytest.raises(KeyError):
        index.node("missing")
    with pytest.raises(KeyError):
        index.descendants("missing")
