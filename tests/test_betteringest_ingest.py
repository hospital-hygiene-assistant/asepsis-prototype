"""
Deterministic unit tests for the BetterIngest PDF ingest integration:
the massager (pins, asset leaves, id join) and pageindex's pin handling.
No OCR, no Ollama, no PDFs — blocks and markdown are synthesised.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pageindex as pi
from modules.ingest._betteringest.betteringest import Asset, IngestedDoc
from modules.ingest._betteringest.ocr import Block
from modules.ingest._massage import massage
from modules.registry import discover


def _fake_doc(tmp_path) -> IngestedDoc:
    """A small two-section document with one figure, mirroring the shapes
    BetterIngest.ingest produces (markdown from to_markdown, blocks from
    layout OCR, assets from save_asset_crops + reachability)."""
    crop = tmp_path / "assets" / "figure_1.png"
    crop.parent.mkdir(parents=True, exist_ok=True)
    crop.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    blocks = [
        Block(label="doc_title", text="Demo Guideline", page=0, bbox=(50, 40, 500, 80)),
        Block(label="paragraph_title", text="Diagnosis", page=0, bbox=(50, 120, 300, 150)),
        Block(label="text", text="How to diagnose.", page=0, bbox=(50, 160, 500, 400)),
        Block(label="paragraph_title", text="Treatment", page=1, bbox=(50, 100, 300, 130)),
        Block(label="text", text="First line therapy.", page=1, bbox=(50, 140, 500, 300)),
        Block(label="figure_title", text="Figure 1: Dosing chart", page=1,
              bbox=(50, 320, 400, 350)),
    ]
    markdown = "\n".join([
        "# Demo Guideline\n",
        "## Diagnosis\n",
        "How to diagnose.\n",
        "## Treatment\n",
        "First line therapy.\n",
        "![figure 1](assets/figure_1.png)\n",
    ])
    assets = [Asset(asset_id="figure_1", type="figure", number=1,
                    caption="Figure 1: Dosing chart", page=2,
                    image=str(crop), sections=["treatment"],
                    description="A dosing chart plotting dose against weight.",
                    bbox=[50.0, 320.0, 420.0, 600.0])]
    return IngestedDoc(doc_name="demo", title="Demo Guideline",
                       pdf_path="/nonexistent/demo.pdf", md_path="",
                       markdown=markdown, assets=assets, tree=None,
                       blocks=blocks, ladder_diag={}, ocr_scale=2.0)


def test_massage_is_deterministic(tmp_path):
    doc = _fake_doc(tmp_path)
    out1 = massage(doc, "demo", "/assets/demo")
    out2 = massage(doc, "demo", "/assets/demo")
    assert out1 == out2


def test_pin_ids_join_with_pageindex_node_ids(tmp_path):
    out = massage(_fake_doc(tmp_path), "demo", "/assets/demo")
    lines = out.split("\n")
    headings = pi._parse_headings(out)
    nodes = pi._build_tree(headings)
    pi._attach_pins(nodes, lines)

    def walk(ns):
        for n in ns:
            yield n
            yield from walk(n.children)

    all_nodes = list(walk(nodes))
    pinned = [n for n in all_nodes if n.pin]
    assert pinned, "massaged markdown must carry pins"
    for n in pinned:
        assert n.pin["id"] == n.node_id            # the join key
        assert n.pin["doc"] == "demo"


def test_asset_becomes_extra_leaf_with_caption_content(tmp_path):
    out = massage(_fake_doc(tmp_path), "demo", "/assets/demo")
    lines = out.split("\n")
    headings = pi._parse_headings(out)
    nodes = pi._build_tree(headings)
    pi._attach_pins(nodes, lines)
    pi._promote_preambles(nodes, lines)
    pi._populate_content(nodes, lines)

    def find(ns, pred):
        for n in ns:
            if pred(n):
                return n
            hit = find(n.children, pred)
            if hit:
                return hit
        return None

    asset = find(nodes, lambda n: (n.pin or {}).get("kind") == "asset")
    assert asset is not None and asset.is_leaf
    # Caption + description feed the RAG decision — never the pin YAML or the
    # raw image link.
    assert "Dosing chart" in asset.content
    assert "```" not in asset.content and "![" not in asset.content
    assert asset.pin["image"] == "/assets/demo/figure_1.png"
    assert asset.pin["page"] == 2 and asset.pin["bbox"]
    # The asset leaf hangs under its citing section (extra leaf convention).
    treatment = find(nodes, lambda n: n.node_id == "treatment")
    assert asset in list(treatment.children) or any(
        asset in c.children for c in treatment.children)


def test_section_pin_carries_page_bbox_regions(tmp_path):
    out = massage(_fake_doc(tmp_path), "demo", "/assets/demo")
    lines = out.split("\n")
    nodes = pi._build_tree(pi._parse_headings(out))
    pi._attach_pins(nodes, lines)

    def find(ns, node_id):
        for n in ns:
            if n.node_id == node_id:
                return n
            hit = find(n.children, node_id)
            if hit:
                return hit
        return None

    treatment = find(nodes, "treatment")
    assert treatment.pin["page"] == 2               # 1-based page
    assert treatment.pin["bbox"] == [50.0, 100.0, 300.0, 130.0]
    assert treatment.pin["regions"] == [[2, 50.0, 140.0, 500.0, 300.0]]
    assert treatment.pin["scale"] == 2.0


def test_plain_markdown_docs_are_unaffected():
    text = "# T\n\n## A\n\nBody text.\n"
    lines = text.split("\n")
    nodes = pi._build_tree(pi._parse_headings(text))
    pi._attach_pins(nodes, lines)
    pi._populate_content(nodes, lines)

    assert nodes[0].pin is None
    leaf = nodes[0].children[0]
    assert leaf.pin is None and leaf.content == "Body text."


def test_strip_pins_removes_blocks_and_images():
    raw = ("```pin\nid: x\nkind: section\n```\n\nReal content.\n\n"
           "![figure 1](/assets/d/f.png)\n\nMore.")
    assert pi._strip_pins(raw) == "Real content.\n\nMore."


def test_registry_discovers_betteringest_pdf():
    mods = discover()
    assert "betteringest_pdf" in mods["ingest"]
    assert mods["ingest"]["betteringest_pdf"]["source"] == "pdf_folder"
    assert "basic_markdown" in mods["ingest"]       # old module still there


def test_save_asset_crops_unlabeled_assets(tmp_path):
    from unittest.mock import MagicMock, patch
    from PIL import Image
    from modules.ingest._betteringest.reconstruct import save_asset_crops
    from modules.ingest._betteringest.ocr import Block

    with patch('pypdfium2.PdfDocument') as mock_pdf_doc:
        mock_page = MagicMock()
        mock_render = MagicMock()
        mock_pil = MagicMock(spec=Image.Image)
        mock_pil.height = 1000
        mock_pil.crop.return_value = mock_pil
        mock_render.to_pil.return_value = mock_pil
        mock_page.render.return_value = mock_render
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_pdf_doc.return_value = mock_doc

        blocks = [
            Block(label="paragraph_title", text="Section 1", page=0, bbox=(50, 100, 300, 130)),
            Block(label="image", text="", page=0, bbox=(50, 200, 400, 400)), # unlabeled figure
            Block(label="table", text="", page=0, bbox=(50, 500, 400, 700)), # unlabeled table
        ]

        manifest = save_asset_crops(blocks, "dummy.pdf", tmp_path / "assets", 2.0)
        assert len(manifest) == 2

        fig = [m for m in manifest if m["type"] == "figure"][0]
        tab = [m for m in manifest if m["type"] == "table"][0]

        assert fig["caption"] == "Figure (unlabeled)"
        assert fig["physical_section"] == "Section 1"
        assert tab["caption"] == "Table (unlabeled)"
        assert tab["physical_section"] == "Section 1"


def test_massage_outputs_table_markdown(tmp_path):
    crop = tmp_path / "assets" / "table_1.png"
    crop.parent.mkdir(parents=True, exist_ok=True)
    crop.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    markdown = "# Doc\n![table 1](assets/table_1.png)\n"
    assets = [Asset(
        asset_id="table_1", type="table", number=1,
        caption="Table 1", page=1, image=str(crop),
        sections=["doc"], description="A desc",
        bbox=[0.0, 0.0, 100.0, 100.0],
        table_markdown="| A | B |\n|---|---|\n| 1 | 2 |"
    )]
    doc = IngestedDoc(
        doc_name="t", title="Doc", pdf_path="/dummy.pdf",
        md_path="", markdown=markdown, assets=assets,
        tree=None, blocks=[], ladder_diag={}, ocr_scale=2.0
    )
    out = massage(doc, "t", "/assets/t")
    # Verify the table image link is replaced by the markdown table
    assert "| A | B |" in out
    assert "![table 1]" not in out


def test_describe_assets_table_prompt_routing(tmp_path):
    from unittest.mock import MagicMock
    from modules.ingest._betteringest.betteringest import BetterIngest

    crop = tmp_path / "assets" / "table_1.png"
    crop.parent.mkdir(parents=True, exist_ok=True)
    crop.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    assets = [
        Asset(
            asset_id="table_1", type="table", number=1,
            caption="Table 1", page=1, image=str(crop),
            table_markdown="| H1 | H2 |\n|---|---|\n| V1 | V2 |"
        )
    ]
    doc = IngestedDoc(
        doc_name="t", title="Doc", pdf_path="/dummy.pdf",
        md_path="", markdown="", assets=assets,
        tree=None, blocks=[], ladder_diag={}, ocr_scale=2.0
    )
    
    chat_mock = MagicMock(return_value="This table shows columns H1 and H2.")
    bi = BetterIngest(out_dir=tmp_path, cache_dir=tmp_path)
    bi.describe_assets(doc, chat=chat_mock)

    # Verify we sent only a text-based prompt (no images list)
    chat_mock.assert_called_once()
    messages = chat_mock.call_args[0][0]
    assert len(messages) == 1
    assert "images" not in messages[0]
    assert "| H1 | H2 |" in messages[0]["content"]


def test_breadcrumb_naming(tmp_path):
    from modules.ingest._betteringest.tree import Node
    from modules.ingest._betteringest.betteringest import find_breadcrumb

    tree = Node(title="Doc Root", children=[
        Node(title="Chapter 1", children=[
            Node(title="Section A", children=[])
        ])
    ])

    breadcrumb = find_breadcrumb(tree, "section a")
    assert breadcrumb == ["Doc Root", "Chapter 1", "Section A"]

