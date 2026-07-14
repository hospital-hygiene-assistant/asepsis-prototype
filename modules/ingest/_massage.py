"""
The "massager" — post-processes BetterIngest's structure-faithful markdown
into asepsis knowledge_base/ markdown with reliable provenance.

What it adds, all deterministic across re-runs of the same folder:

  · one ```pin fenced block (inline YAML, in-body — never frontmatter) right
    under every heading, carrying the page / bbox / body-text regions of that
    section in the source PDF.  Locations come straight from the OCR layout
    blocks (Block.bbox, cached by BetterIngester) — nothing is re-extracted.
  · pin ids that are byte-identical to the node_ids asepsis's own indexer
    (pageindex._parse_headings) will assign to the same headings — the pin id
    IS the join key between RAG chunks, the doc viewer, and the PDF.
  · assets as "extra leaves": each figure/table image link BetterIngest placed
    under its citing section is expanded into a subsection whose heading is
    the asset's caption (extending IngestedDoc.assets' citing-section
    convention).  The leaf body is the caption + the multimodal description —
    that text (not the raw crop) is what the RAG decision step reads; the pin
    still resolves caption → crop image → PDF location.

The pin blocks are ordinary fenced code blocks (info string "pin", YAML body),
so any markdown renderer shows them and one CSS toggle hides them — a single
render path.  asepsis's pageindex strips them from LLM-visible content and
lifts them into structured node metadata.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ._betteringest.anchors import heading_pages_from_blocks  # noqa: F401  (convention ref)
from ._betteringest.betteringest import Asset, IngestedDoc
from ._betteringest.reconstruct import reading_order
from ._betteringest.tree import normalise_title

# BetterIngest emits exactly this shape for asset links (see betteringest.py):
#   ![figure 1](assets/figure_1.png)
_ASSET_LINK_RE = re.compile(r"^!\[(figure|table) (\d+)\]\((assets/[^)]+)\)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

_UNPLACED_HEADING = "## Figures and Tables"


def _fmt_num(v: float) -> float:
    return round(float(v), 1)


def _fmt_bbox(bbox) -> str:
    return "[" + ", ".join(str(_fmt_num(v)) for v in bbox) + "]"


def _heading_provenance(doc: IngestedDoc) -> list[dict]:
    """Per detected heading (reading order): its page/bbox plus the per-page
    union bbox of the body-text blocks under it — straight from the layout
    Blocks, mirroring how build_content_tree partitions content."""
    entries: list[dict] = []
    current: dict | None = None
    for b in reading_order(doc.blocks):
        if b.label in ("doc_title", "paragraph_title") and b.text.strip():
            current = {
                "norm": normalise_title(b.text),
                "page": int(b.page) + 1,               # 1-based, PageIndex convention
                "bbox": list(b.bbox),
                "regions": {},                          # page -> union bbox
            }
            entries.append(current)
        elif b.label == "text" and current is not None:
            pg = int(b.page) + 1
            x0, y0, x1, y1 = b.bbox
            r = current["regions"].get(pg)
            current["regions"][pg] = ([x0, y0, x1, y1] if r is None else
                                      [min(r[0], x0), min(r[1], y0),
                                       max(r[2], x1), max(r[3], y1)])
    return entries


def _pin_block(fields: list[tuple[str, str]]) -> list[str]:
    lines = ["```pin"]
    lines += [f"{k}: {v}" for k, v in fields]
    lines += ["```", ""]
    return lines


def _section_pin(node_id: str, stem: str, prov: dict | None,
                 scale: float) -> list[str]:
    fields: list[tuple[str, str]] = [
        ("id", node_id), ("kind", "section"), ("doc", stem)]
    if prov is not None:
        regions = [[pg] + [_fmt_num(v) for v in bb]
                   for pg, bb in sorted(prov["regions"].items())]
        fields += [("page", str(prov["page"])),
                   ("bbox", _fmt_bbox(prov["bbox"]))]
        if regions:
            fields.append(("regions", json.dumps(regions)))
    fields.append(("scale", str(scale)))
    return _pin_block(fields)


def _asset_pin(node_id: str, stem: str, a: Asset, image_url: str,
               scale: float) -> list[str]:
    fields = [("id", node_id), ("kind", "asset"), ("doc", stem),
              ("asset", a.asset_id), ("type", a.type),
              ("page", str(a.page))]
    if a.bbox:
        fields.append(("bbox", _fmt_bbox(a.bbox)))
    fields += [("image", image_url), ("scale", str(scale))]
    return _pin_block(fields)


def _caption_heading(caption: str, max_len: int = 120) -> str:
    """Asset-leaf heading text: the caption, single-line, markdown-safe."""
    text = " ".join(caption.split()).replace("#", "").strip()
    return text if len(text) <= max_len else text[:max_len - 1].rstrip() + "…"


def massage(doc: IngestedDoc, stem: str, asset_url_base: str) -> str:
    """BetterIngest markdown → knowledge_base markdown with pins.

    `asset_url_base` is the URL prefix the app serves this document's asset
    crops under (e.g. "/assets/<stem>").  Returns the massaged markdown.
    """
    # pageindex lives at the asepsis root (already on sys.path via the
    # module registry).  Using ITS heading parser guarantees pin ids match the
    # node_ids the index stage will assign — same slugs, same dedup order.
    from pageindex.nodes import _parse_headings

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
    headings = _parse_headings(text)               # [(line_idx, level, title, node_id)]

    prov_queue = _heading_provenance(doc)
    pins_at: dict[int, list[str]] = {}
    for line_idx, level, title, node_id in headings:
        if line_idx in asset_at_line:
            a = asset_at_line[line_idx]
            pins_at[line_idx] = _asset_pin(
                node_id, stem, a, f"{asset_url_base}/{Path(a.image).name}",
                doc.ocr_scale)
            continue
        # Queue-consuming title match — the attach_page_anchors convention
        # (anchors.py), so repeated titles resolve by position.
        key = normalise_title(title)
        prov = None
        for i, entry in enumerate(prov_queue):
            if entry["norm"] == key:
                prov = entry
                del prov_queue[:i + 1]
                break
        pins_at[line_idx] = _section_pin(node_id, stem, prov, doc.ocr_scale)

    lines = text.split("\n")
    for line_idx in sorted(pins_at, reverse=True):
        insert_at = line_idx + 1
        # keep the blank line that follows a heading before the pin block
        if insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1
        lines[insert_at:insert_at] = pins_at[line_idx]

    return "\n".join(lines)
