"""Clinical provenance from PDF ingest to an exact visual location."""

import pytest

from pageindex.pins import (
    AssetProvenance,
    PixelBox,
    PinValidationError,
    ProvenancePin,
    SourceSpan,
    emit_pin,
    locate_visual_citation,
    parse_pin,
)
from pageindex.nodes import parse_document

from tests.test_betteringest_ingest import _fake_doc
from modules.ingest._betteringest.ocr import Block
from modules.ingest._massage import massage


def test_a_provenance_pin_survives_its_fenced_representation():
    pin = ProvenancePin(
        version=2,
        document="mrsa-guideline",
        node_id="isolation",
        spans=(
            SourceSpan(
                page=3,
                start=0,
                end=22,
                box=PixelBox(50.0, 200.0, 500.0, 460.0),
            ),
        ),
        scale=2.0,
    )

    assert parse_pin(emit_pin(pin)) == pin


def test_one_exact_quote_crossing_pages_locates_every_source_region():
    content = "Wear gloves. Use gown."
    pin = ProvenancePin(
        version=2,
        document="mrsa-guideline",
        node_id="ppe",
        spans=(
            SourceSpan(1, 0, 12, PixelBox(100, 200, 600, 400)),
            SourceSpan(2, 13, 22, PixelBox(50, 100, 450, 300)),
        ),
        scale=2.0,
    )

    location = locate_visual_citation(
        pin,
        content,
        "gloves. Use",
        page_size=lambda _document, _page: (1000.0, 2000.0),
    )

    assert location.status == "exact"
    assert [region.page for region in location.regions] == [1, 2]
    assert location.regions[0].x == 0.1
    assert location.regions[0].y == 0.1
    assert location.regions[0].width == 0.5
    assert location.regions[0].height == 0.1


def test_a_repeated_quote_never_produces_a_highlight():
    pin = ProvenancePin(
        version=2,
        document="mrsa-guideline",
        node_id="ppe",
        spans=(SourceSpan(1, 0, 23, PixelBox(10, 20, 100, 80)),),
        scale=2.0,
    )

    location = locate_visual_citation(
        pin,
        "Wear gloves. Wear gloves.",
        "Wear gloves.",
        page_size=lambda _document, _page: (1000.0, 2000.0),
    )

    assert location.status == "unavailable"
    assert location.regions == ()
    assert location.reason == "ambiguous_quote"


def test_an_asset_uses_its_source_region_directly():
    pin = ProvenancePin(
        version=2,
        document="mrsa-guideline",
        node_id="figure-1",
        spans=(SourceSpan(2, 0, 44, PixelBox(50, 320, 420, 600)),),
        scale=2.0,
        asset=AssetProvenance(
            asset_id="figure_1",
            asset_type="figure",
            image="/assets/mrsa-guideline/figure_1.png",
        ),
    )

    location = locate_visual_citation(
        pin,
        "Figure 1: PPE sequence. Gloves precede gown.",
        "",
        page_size=lambda _document, _page: (1000.0, 2000.0),
    )

    assert location.status == "exact"
    assert [region.page for region in location.regions] == [2]


def test_real_ingest_to_pageindex_locates_the_quoted_body_not_its_heading(tmp_path):
    markdown = massage(_fake_doc(tmp_path), "demo", "/assets/demo")
    nodes = parse_document(markdown)

    def walk(items):
        for item in items:
            yield item
            yield from walk(item.children)

    treatment_overview = next(
        node for node in walk(nodes)
        if node.synthetic and node.parent_title == "Treatment"
    )
    location = locate_visual_citation(
        treatment_overview.pin,
        treatment_overview.content,
        "First line therapy.",
        page_size=lambda _document, _page: (1000.0, 1000.0),
    )

    assert location.status == "exact"
    assert len(location.regions) == 1
    assert location.regions[0].page == 2
    assert location.regions[0].y == 0.14


def test_pageindex_rejects_a_pin_attached_to_the_wrong_heading():
    pin = ProvenancePin(
        version=2,
        document="guide",
        node_id="different-heading",
        spans=(SourceSpan(1, 0, 4, PixelBox(10, 20, 100, 80)),),
        scale=2.0,
    )
    markdown = f"# Actual Heading\n\n{emit_pin(pin)}\nBody"

    try:
        parse_document(markdown)
    except PinValidationError as exc:
        assert "different-heading" in str(exc)
        assert "actual-heading" in str(exc)
    else:
        raise AssertionError("a mismatched provenance identity was accepted")


def test_real_ingest_preserves_every_page_crossed_by_a_quote(tmp_path):
    doc = _fake_doc(tmp_path)
    doc.blocks.append(Block(
        label="text",
        text="Continue precautions.",
        page=2,
        bbox=(60, 100, 520, 280),
    ))
    doc.markdown = doc.markdown.replace(
        "First line therapy.",
        "First line therapy. Continue precautions.",
    )
    nodes = parse_document(massage(doc, "demo", "/assets/demo"))

    def walk(items):
        for item in items:
            yield item
            yield from walk(item.children)

    treatment_overview = next(
        node for node in walk(nodes)
        if node.synthetic and node.parent_title == "Treatment"
    )
    location = locate_visual_citation(
        treatment_overview.pin,
        treatment_overview.content,
        "therapy. Continue",
        page_size=lambda _document, _page: (1000.0, 1000.0),
    )

    assert location.status == "exact"
    assert [region.page for region in location.regions] == [2, 3]


def test_a_quote_missing_from_canonical_content_has_no_highlight():
    pin = ProvenancePin(
        2,
        "guide",
        "ppe",
        (SourceSpan(1, 0, 12, PixelBox(10, 20, 100, 80)),),
        2.0,
    )

    location = locate_visual_citation(
        pin,
        "Wear gloves.",
        "Wear a respirator.",
        page_size=lambda _document, _page: (1000.0, 2000.0),
    )

    assert location.status == "unavailable"
    assert location.regions == ()
    assert location.reason == "quote_not_found"


def test_an_unavailable_pdf_page_has_no_highlight():
    pin = ProvenancePin(
        2,
        "guide",
        "ppe",
        (SourceSpan(1, 0, 12, PixelBox(10, 20, 100, 80)),),
        2.0,
    )

    location = locate_visual_citation(
        pin,
        "Wear gloves.",
        "Wear gloves.",
        page_size=lambda _document, _page: None,
    )

    assert location.status == "unavailable"
    assert location.regions == ()
    assert location.reason == "page_size_unavailable"


def test_a_source_span_outside_canonical_content_has_no_highlight():
    pin = ProvenancePin(
        2,
        "guide",
        "ppe",
        (SourceSpan(1, 0, 999, PixelBox(10, 20, 100, 80)),),
        2.0,
    )

    location = locate_visual_citation(
        pin,
        "Wear gloves.",
        "Wear gloves.",
        page_size=lambda _document, _page: (1000.0, 2000.0),
    )

    assert location.status == "unavailable"
    assert location.regions == ()
    assert location.reason == "invalid_source_span"


def test_a_partially_mapped_quote_has_no_highlight():
    content = "Wear gloves."
    pin = ProvenancePin(
        2,
        "guide",
        "ppe",
        (SourceSpan(1, 5, 11, PixelBox(10, 20, 100, 80)),),
        2.0,
    )

    location = locate_visual_citation(
        pin,
        content,
        content,
        page_size=lambda _document, _page: (1000.0, 2000.0),
    )

    assert location.status == "unavailable"
    assert location.regions == ()
    assert location.reason == "quote_not_fully_mapped"


@pytest.mark.parametrize(
    "raw",
    [
        '{"version":2.9,"document":"d","nodeId":"n","spans":[],"scale":2}',
        '{"version":2,"document":null,"nodeId":"n","spans":[],"scale":2}',
        '{"version":2,"document":"d","nodeId":"n","spans":[{"page":2.9,"start":0,"end":4,"box":[1,2,3,4]}],"scale":2}',
        '{"version":2,"document":"d","nodeId":"n","spans":[{"page":1,"start":0.9,"end":4,"box":[1,2,3,4]}],"scale":2}',
    ],
)
def test_pin_parser_rejects_coercible_but_inexact_values(raw):
    with pytest.raises(PinValidationError):
        parse_pin(raw)
