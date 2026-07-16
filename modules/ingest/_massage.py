"""
The "massager" — post-processes BetterIngest's structure-faithful markdown
into asepsis knowledge_base/ markdown with reliable provenance.

What it adds, all deterministic across re-runs of the same folder:

  · one versioned ```pin fenced block (in-body — never frontmatter) right
    under every heading, mapping exact canonical-text character spans to their
    source-PDF regions. Locations come straight from cached OCR layout blocks;
    blocks that do not align verbatim are omitted rather than guessed.
  · pin ids that are byte-identical to the node_ids asepsis's own public
    heading identity interface will assign to the same headings — the pin id
    IS the join key between RAG chunks, the doc viewer, and the PDF.
  · assets as "extra leaves": each figure/table image link BetterIngest placed
    under its citing section is expanded into a subsection whose heading is
    the asset's caption (extending IngestedDoc.assets' citing-section
    convention).  The leaf body is the caption + the multimodal description —
    that text (not the raw crop) is what the RAG decision step reads; the pin
    still resolves caption → crop image → PDF location.

The pin blocks are ordinary fenced code blocks (info string "pin", JSON body),
so any markdown renderer shows them and one CSS toggle hides them — a single
render path.  asepsis's pageindex strips them from LLM-visible content and
lifts them into structured node metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pageindex.nodes import HeadingIdentity, heading_identities
from pageindex.pins import (
    AssetProvenance,
    PixelBox,
    ProvenancePin,
    SourceSpan,
    emit_pin,
    strip_provenance_markup,
)

from ._betteringest.anchors import heading_pages_from_blocks  # noqa: F401  (convention ref)
from ._betteringest.betteringest import Asset, IngestedDoc
from ._betteringest.ocr import Block
from ._betteringest.reconstruct import reading_order
from ._betteringest.tree import normalise_title

# BetterIngest emits exactly this shape for asset links (see betteringest.py):
#   ![figure 1](assets/figure_1.png)
_ASSET_LINK_RE = re.compile(r"^!\[(figure|table) (\d+)\]\((assets/[^)]+)\)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

_UNPLACED_HEADING = "## Figures and Tables"


@dataclass
class _HeadingProvenance:
    normalized_title: str
    blocks: list[Block] = field(default_factory=list)


def _heading_provenance(doc: IngestedDoc) -> list[_HeadingProvenance]:
    """Body blocks grouped under each detected heading in reading order."""
    entries: list[_HeadingProvenance] = []
    current: _HeadingProvenance | None = None
    for b in reading_order(doc.blocks):
        if b.label in ("doc_title", "paragraph_title") and b.text.strip():
            current = _HeadingProvenance(normalise_title(b.text))
            entries.append(current)
        elif b.label == "text" and current is not None:
            current.blocks.append(b)
    return entries


def _canonical_body(
    lines: list[str],
    heading: HeadingIdentity,
    next_heading: HeadingIdentity | None,
) -> str:
    """The exact immediate body text PageIndex will attach to this heading."""
    end = next_heading.line_idx if next_heading is not None else len(lines)
    return strip_provenance_markup("\n".join(lines[heading.line_idx + 1:end]).strip())


def _source_spans(content: str, blocks: list[Block]) -> tuple[SourceSpan, ...]:
    """Map only verbatim body blocks into canonical PageIndex character space.

    PDF text extraction and OCR text can differ. A block that cannot be aligned
    exactly is deliberately omitted; the locator will then decline to draw a
    highlight rather than guess.
    """
    spans: list[SourceSpan] = []
    cursor = 0
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        start = content.find(text, cursor)
        if start < 0:
            continue
        try:
            box = PixelBox(*(round(float(value), 1) for value in block.bbox))
            spans.append(SourceSpan(
                page=int(block.page) + 1,
                start=start,
                end=start + len(text),
                box=box,
            ))
        except (TypeError, ValueError):
            continue
        cursor = start + len(text)
    return tuple(spans)


def _pin_lines(pin: ProvenancePin) -> list[str]:
    return [*emit_pin(pin).splitlines(), ""]


def _section_pin(
    node_id: str,
    stem: str,
    content: str,
    prov: _HeadingProvenance | None,
    scale: float,
) -> list[str]:
    spans = _source_spans(content, prov.blocks if prov is not None else [])
    return _pin_lines(ProvenancePin(2, stem, node_id, spans, scale))


def _asset_pin(
    node_id: str,
    stem: str,
    content: str,
    asset: Asset,
    image_url: str,
    scale: float,
) -> list[str]:
    spans: tuple[SourceSpan, ...] = ()
    if asset.bbox and content:
        try:
            spans = (SourceSpan(
                page=int(asset.page),
                start=0,
                end=len(content),
                box=PixelBox(*(round(float(value), 1) for value in asset.bbox)),
            ),)
        except (TypeError, ValueError):
            pass
    pin = ProvenancePin(
        version=2,
        document=stem,
        node_id=node_id,
        spans=spans,
        scale=scale,
        asset=AssetProvenance(asset.asset_id, asset.type, image_url),
    )
    return _pin_lines(pin)


def _caption_heading(caption: str, max_len: int = 120) -> str:
    """Asset-leaf heading text: the caption, single-line, markdown-safe."""
    text = " ".join(caption.split()).replace("#", "").strip()
    return text if len(text) <= max_len else text[:max_len - 1].rstrip() + "…"


def massage(doc: IngestedDoc, stem: str, asset_url_base: str) -> str:
    """BetterIngest markdown → knowledge_base markdown with pins.

    `asset_url_base` is the URL prefix the app serves this document's asset
    crops under (e.g. "/assets/<stem>").  Returns the massaged markdown.
    """
    assets_by_relpath = {
        f"assets/{Path(a.image).name}": a for a in doc.assets}

    # Pass 1 — expand every asset image link into an asset leaf subsection
    # under the section it sits in (its citing section, per BetterIngest's own
    # placement), and drop nothing else.
    out_lines: list[str] = []
    pending_assets: list[tuple[int, Asset]] = []   # (heading line idx in out, asset)
    current_level = 1
    for line in doc.markdown.split("\n"):
        hm = _HEADING_RE.match(line)
        if hm:
            current_level = len(hm.group(1))
            out_lines.append(line)
            continue
        am = _ASSET_LINK_RE.match(line)
        if am and am.group(3) in assets_by_relpath:
            a = assets_by_relpath[am.group(3)]
            level = min(current_level + 1, 6)
            heading = f"{'#' * level} {_caption_heading(a.caption)}"
            body = " ".join(x for x in (a.caption.strip(), a.description.strip())
                            if x)
            image_url = f"{asset_url_base}/{Path(a.image).name}"
            pending_assets.append((len(out_lines), a))
            out_lines += [heading, "",
                          f"![{a.type} {a.number}]({image_url})", "",
                          body, ""]
            continue
        out_lines.append(line)

    asset_at_line = {idx: a for idx, a in pending_assets}

    # Pass 2 — compute node ids exactly as the indexer will, then insert one
    # pin block right under each heading (reverse order keeps indices valid).
    text = "\n".join(out_lines)
    headings = heading_identities(text)
    source_lines = text.split("\n")

    prov_queue = _heading_provenance(doc)
    pins_at: dict[int, list[str]] = {}
    for index, heading in enumerate(headings):
        next_heading = headings[index + 1] if index + 1 < len(headings) else None
        content = _canonical_body(source_lines, heading, next_heading)
        if heading.line_idx in asset_at_line:
            a = asset_at_line[heading.line_idx]
            pins_at[heading.line_idx] = _asset_pin(
                heading.node_id, stem, content, a,
                f"{asset_url_base}/{Path(a.image).name}", doc.ocr_scale)
            continue
        # Queue-consuming title match — the attach_page_anchors convention
        # (anchors.py), so repeated titles resolve by position.
        key = normalise_title(heading.title)
        prov = None
        for i, entry in enumerate(prov_queue):
            if entry.normalized_title == key:
                prov = entry
                del prov_queue[:i + 1]
                break
        pins_at[heading.line_idx] = _section_pin(
            heading.node_id, stem, content, prov, doc.ocr_scale)

    lines = text.split("\n")
    for line_idx in sorted(pins_at, reverse=True):
        insert_at = line_idx + 1
        # keep the blank line that follows a heading before the pin block
        if insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1
        lines[insert_at:insert_at] = pins_at[line_idx]

    return "\n".join(lines)
