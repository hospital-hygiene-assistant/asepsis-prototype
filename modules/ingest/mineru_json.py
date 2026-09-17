"""
Module 1 — Ingest: MinerU annotation sessions

Turns an annotation session (MinerU layout detection + a human pass over the
region types and heading levels) into astepsis knowledge_base/ markdown with
the same provenance pins the BetterIngest path emits.

Where betteringest_pdf RECONSTRUCTS structure from a PDF (layout OCR →
escalation ladder → markdown), this module is handed the structure already
settled: `final_state.json` carries every region's type, bbox, reading order
and — once annotated — its heading level.  So the whole OCR stack drops out:
no paddle, no paddlex, no numpy, no ladder, no font-size level assignment.

What the session gives us, and what it does not:

  · types + heading levels + reading order + bboxes  →  the document tree
  · `metadata.html` on every table region             →  markdown pipe tables
  · NOTHING in `text` (MinerU's model.json carries layout boxes, not content)

That last one is why `source.pdf` is required rather than optional: body text
comes from the PDF text layer, read per-region with BetterIngest's own
`block_text` so hyphenation and soft-hyphen handling match the other path.
A scanned PDF has no text layer and is reported as a warning, never silently
ingested empty — this module has no OCR fallback by design.

Coordinates: session bboxes are fractions of the page (0-1).  Pins are render
pixels at `OCR_SCALE`, top-left origin — the convention `/api/document/{stem}/
page/{n}` draws in, and it reads the scale back from `.sources.json`, so the
two must agree.  `_to_pixels` is the only place that conversion happens.

The source PDF is COPIED into knowledge_base/sources/ and pins point at the
copy: a session folder that later moves or is cleaned up would otherwise take
every pin in the document down with it.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from statistics import median

MODULE_INFO = {
    "stage": "ingest",
    "name": "mineru_json",
    "label": "MinerU (annotated)",
    "description": (
        "Ingests annotation sessions (MinerU layout + a human pass over region "
        "types and heading levels) into knowledge_base/. Heading hierarchy "
        "comes from the annotation, body text from the PDF text layer, tables "
        "from the session's own HTML. Emits the same provenance pins as the "
        "BetterIngest path. Needs no OCR stack."),
    # Reuses the folder-picker flow: the chosen folder is scanned (recursively)
    # for annotation sessions.
    "source": "pdf_folder",
}

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "knowledge_base"
PDF_DIR = KB_DIR / "sources"                 # stable home for the source PDFs
OUT_DIR = ROOT / ".mineru_out"               # crops + caption cache
CONFIG_PATH = ROOT / ".mineru.json"
SOURCES_MANIFEST = KB_DIR / ".sources.json"
# {doc_id: {page: [label, ...]}} — see evaluation/golden_from_csv.py.
GOLDEN_PATH = KB_DIR / ".golden.json"

STATE_NAME = "final_state.json"
PDF_NAME = "source.pdf"

# Pins are render pixels at this scale; `.sources.json` carries it so the page
# renderer re-renders at exactly the same scale.  Matches the other path.
OCR_SCALE = 2.0

# ── region taxonomy (MinerU's model.json labels, as the session preserves them)

HEADING_TYPES = {"doc_title", "paragraph_title"}
# Everything that is prose belonging to the current section.  Deliberately
# wider than the BetterIngest path's bare "text": the session distinguishes
# abstracts, reference lists and footnotes, and all of them are body.
BODY_TYPES = {"text", "abstract", "content", "reference_content",
              "aside_text", "footnote", "vision_footnote"}
ASSET_TYPES = {"table": "table", "image": "figure"}
CAPTION_TYPES = {"figure_title", "table_caption", "image_caption"}
# Running heads, folios and the line-level OCR boxes that sit inside the
# layout blocks — none of them are content, and unioning them into a section's
# body region would stretch it to the full page.
#
# "header" is in here only as a DEFAULT. Annotators retype section headings as
# `header`, so the type alone cannot decide: what makes a region a heading is
# carrying a `heading_level`, whatever its type says.  A header without one is
# a running head and is dropped; a header with one is a heading.
DROPPED_TYPES = {"ocr_text", "header", "footer", "number", "footer_image"}
# Used only for a session nobody annotated (see `load_regions`).
DEFAULT_LEVEL = {"doc_title": 1, "paragraph_title": 2}
MAX_LEVEL = 6                                # markdown (and HEADING_RE) cap

# Layout OCR detects each pictogram in an infographic as its own figure, so a
# priority-group graphic becomes fifteen "figures" with hallucinated captions.
# Size alone cannot identify them — measured on the BPPL report, the icons
# score 0.78-0.96 detection confidence while genuine small charts score
# 0.52-0.73, so a size or confidence cut removes real figures and keeps the
# icons. What does identify them is REPETITION: a run of same-sized graphics
# on one page, none of which carries a caption, is a grid of decorations.
ICON_CLUSTER_MIN = 6            # fewer than this on a page stays manageable
ICON_SIZE_TOLERANCE = 0.20      # same-sized = both dims within +/-20% of the cluster
ICON_GROUP_HEADING = "Likely icons/broken infographics"


# ── source-folder config ─────────────────────────────────────────────────────

def get_source_dir() -> str | None:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("source_dir")
    except Exception:
        return None


def set_source_dir(path: str) -> None:
    CONFIG_PATH.write_text(json.dumps({"source_dir": str(path)}, indent=2),
                           encoding="utf-8")


def load_golden(stem: str) -> dict[int, list[str]]:
    """This document's golden labels, as {1-based page: [label, ...]}.

    Eval questions name a document and a page, so the labels are stamped onto
    every chunk whose pin covers that page — which is a recall set, not a
    single answer: a page routinely spans several chunks, and which of them
    holds the answer is exactly what the eval is measuring.
    """
    try:
        raw = json.loads(GOLDEN_PATH.read_text(encoding="utf-8")).get(stem, {})
    except Exception:
        return {}
    return {int(page): list(labels) for page, labels in raw.items()}


def _load_sources() -> dict:
    try:
        return json.loads(SOURCES_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_sessions(root: Path) -> list[Path]:
    """Every annotation session under `root` — a folder holding final_state.json.

    Sorted for determinism; hidden folders skipped so caches never get walked.
    """
    if (root / STATE_NAME).is_file():
        return [root]
    out = []
    for state in sorted(root.rglob(STATE_NAME)):
        if any(part.startswith(".") for part in state.relative_to(root).parts):
            continue
        out.append(state.parent)
    return out


# ── the session, as regions ──────────────────────────────────────────────────

@dataclass
class Region:
    type: str
    text: str
    page: int                       # 0-based
    bbox: list[float]               # render px @ OCR_SCALE, top-left origin
    level: int | None = None        # headings only; None means "not a heading"
    order: float | None = None      # per-page, 1-based; None until inferred
    html: str = ""                  # tables only


@dataclass
class _Asset:
    asset_id: str
    type: str                       # "table" | "figure"
    number: int
    caption: str
    page: int                       # 1-based, pin convention
    image: str
    bbox: list[float]
    sections: list[str] = field(default_factory=list)
    description: str = ""
    table_markdown: str = ""
    is_icon: bool = False           # part of an uncaptioned same-size run

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _to_pixels(bbox: dict, page_w: float, page_h: float) -> list[float]:
    """Session bbox (page fractions) → render px at OCR_SCALE.

    Both conventions put the origin top-left, so this is a pure scale — no
    flip.  Rounded to match the pin formatter's own precision.
    """
    return [round(bbox["x0"] * page_w * OCR_SCALE, 1),
            round(bbox["y0"] * page_h * OCR_SCALE, 1),
            round(bbox["x1"] * page_w * OCR_SCALE, 1),
            round(bbox["y1"] * page_h * OCR_SCALE, 1)]


def load_regions(state: dict) -> list[Region]:
    """Session regions → content regions, in reading order.

    Drops what the annotator ignored and what is not content. `reading_order`
    is per-page and 1-based; regions without one (the line-level ocr_text
    boxes) are dropped anyway, so a missing order sorts last rather than
    silently jumping to the front.
    """
    pages = {p["page_index"]: (p["width"], p["height"])
             for p in state["document"]["pages"]}
    regions = state.get("regions", [])
    # An ANNOTATED session is trusted completely: a heading is a region with a
    # level, and a title-typed region without one was left out of the hierarchy
    # deliberately, so it is demoted to body text rather than guessed at level
    # 2.  (MinerU types figure-internal labels — "Critical group", "Sources:" —
    # as paragraph_title; defaulting those to 2 opens a top-level section for
    # each.)  A session nobody annotated has no levels at all, and demoting
    # every title would leave one structureless document — so there, and only
    # there, MinerU's own types still set the level.
    annotated = any(r.get("heading_level") is not None for r in regions)
    out: list[Region] = []
    for r in regions:
        rtype = r.get("type", "")
        if r.get("ignored"):
            continue
        level = None
        if r.get("heading_level") is not None:
            level = max(1, min(int(r["heading_level"]), MAX_LEVEL))
        elif not annotated and rtype in HEADING_TYPES:
            level = DEFAULT_LEVEL.get(rtype, 2)
        if level is None:
            if rtype in DROPPED_TYPES:
                continue
            # Demoted titles keep their TEXT — they are body, not noise.
            if rtype not in BODY_TYPES and rtype not in ASSET_TYPES \
                    and rtype not in CAPTION_TYPES and rtype not in HEADING_TYPES:
                continue
        page = int(r["page"])
        if page not in pages:
            continue
        w, h = pages[page]
        out.append(Region(
            type=rtype,
            text=(r.get("text") or "").strip(),
            page=page,
            bbox=_to_pixels(r["bbox"], w, h),
            level=level,
            order=r.get("reading_order"),
            html=((r.get("metadata") or {}).get("html") or ""),
        ))
    _infer_missing_order(out)
    out.sort(key=lambda r: (r.page, r.order))
    return out


def _infer_missing_order(regions: list[Region]) -> None:
    """Give a reading order to regions that have none, from their position.

    A region the annotator drew by hand has `reading_order: null`, and so does
    one whose order the tool cleared.  Sorting those last would put a heading
    after its own body — the section would then collect the NEXT page's text
    and lose its own.  Instead each one lands just after the last ordered
    region that starts above it, which is where it reads on the page.
    """
    by_page: dict[int, list[Region]] = {}
    for r in regions:
        by_page.setdefault(r.page, []).append(r)
    for page_regions in by_page.values():
        known = sorted(((r.order, r.bbox[1]) for r in page_regions
                        if r.order is not None), key=lambda t: t[0])
        for r in page_regions:
            if r.order is not None:
                continue
            above = [o for o, y0 in known if y0 <= r.bbox[1]]
            r.order = (max(above) if above else 0) + 0.5


def fill_text_from_pdf(regions: list[Region], pdf_path: Path) -> int:
    """Fill every empty region from the PDF text layer. Returns the count that
    stayed empty — pages with no text layer (scanned), which the caller warns
    about rather than ingesting as silent blanks."""
    import pypdfium2 as pdfium

    from modules.ingest._betteringest.ocr import Block
    from modules.ingest._betteringest.reconstruct import block_text

    doc = pdfium.PdfDocument(str(pdf_path))
    pages: dict[int, tuple] = {}
    empty = 0
    for r in regions:
        if r.text or r.type in ASSET_TYPES:
            continue
        if r.page not in pages:
            pg = doc[r.page]
            pages[r.page] = (pg.get_textpage(), pg.get_height())
        textpage, height = pages[r.page]
        r.text = block_text(
            Block(label="text", text="", page=r.page, bbox=tuple(r.bbox)),
            textpage, height, OCR_SCALE)
        if not r.text:
            empty += 1
    return empty


# ── tables: session HTML → markdown ──────────────────────────────────────────

class _TableParser(HTMLParser):
    """Rows of (text, rowspan, colspan). MinerU's table HTML is plain
    tr/td/th with the occasional span — no nesting, no attributes we need."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self._row: list | None = None
        self._cell: list[str] | None = None
        self._span = (1, 1)

    @staticmethod
    def _span_of(attrs: dict, key: str) -> int:
        try:
            return max(1, int(attrs.get(key, 1)))
        except (TypeError, ValueError):
            return 1

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            if self._row is None:
                self._row = []
            self._cell = []
            self._span = (self._span_of(a, "rowspan"), self._span_of(a, "colspan"))

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            text = " ".join("".join(self._cell).split())
            self._row.append((text, *self._span))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def close(self):
        super().close()
        if self._row:                       # unclosed final <tr>
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_markdown(html: str) -> str:
    """MinerU table HTML → a pipe table.

    Spans are expanded by occupancy (the spanned cell keeps its text, the
    cells it covers go blank) because pipe tables cannot express them — a
    reader and the retrieval step both do better with a rectangular grid than
    with raw markup.  Returns "" if there is nothing tabular to render.
    """
    if not html or "<t" not in html:
        return ""
    p = _TableParser()
    try:
        p.feed(html)
        p.close()
    except Exception:
        return ""
    occupied: dict[tuple[int, int], str] = {}
    for ri, row in enumerate(p.rows):
        ci = 0
        for text, rowspan, colspan in row:
            while (ri, ci) in occupied:
                ci += 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    occupied[(ri + dr, ci + dc)] = (
                        text if (dr == 0 and dc == 0) else "")
            ci += colspan
    if not occupied:
        return ""
    rows = max(r for r, _ in occupied) + 1
    cols = max(c for _, c in occupied) + 1
    grid = [[occupied.get((r, c), "").replace("|", r"\|")
             for c in range(cols)] for r in range(rows)]
    lines = ["| " + " | ".join(grid[0]) + " |",
             "| " + " | ".join(["---"] * cols) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in grid[1:]]
    return "\n".join(lines)


# ── assets: crops from the PDF ───────────────────────────────────────────────

def flag_icon_clusters(regions: list[Region]) -> set[int]:
    """Indices of asset regions that look like infographic decoration.

    An asset is flagged when, on its own page, it belongs to a run of at least
    ICON_CLUSTER_MIN same-sized assets that NONE of which carries a caption.
    All three conditions matter: repetition alone would catch a gallery of
    real figures, same-size alone would catch a two-column layout, and a
    caption is the document itself saying the graphic is worth referring to.

    Deliberately not a size threshold. A lone small icon stays — one is not
    annoying, and a genuinely useful small graphic keeps its place.
    """
    by_page: dict[int, list[int]] = {}
    for i, r in enumerate(regions):
        if r.type in ASSET_TYPES:
            by_page.setdefault(r.page, []).append(i)

    flagged: set[int] = set()
    for page_indices in by_page.values():
        sized = sorted(((i, regions[i].bbox[2] - regions[i].bbox[0],
                         regions[i].bbox[3] - regions[i].bbox[1])
                        for i in page_indices), key=lambda t: t[1] * t[2])
        clusters: list[list[tuple]] = []
        for item in sized:
            for cluster in clusters:
                mw = median(c[1] for c in cluster)
                mh = median(c[2] for c in cluster)
                if (abs(item[1] - mw) <= ICON_SIZE_TOLERANCE * mw
                        and abs(item[2] - mh) <= ICON_SIZE_TOLERANCE * mh):
                    cluster.append(item)
                    break
            else:
                clusters.append([item])
        for cluster in clusters:
            uncaptioned = [c[0] for c in cluster
                           if not _caption_for(regions, c[0])]
            if len(uncaptioned) >= ICON_CLUSTER_MIN:
                flagged.update(uncaptioned)
    return flagged


def _crop_assets(regions: list[Region], pdf_path: Path, out_dir: Path,
                 headings_at: dict[int, str],
                 icons: set[int] | None = None) -> list[_Asset]:
    """One asset per asset region — crops rendered from the PDF at the pin's
    own bbox, so the crop and the highlight can never disagree.

    Regions are kept 1:1 with the session: a table continued across pages is
    several regions there and stays several assets here, because merging them
    would mean inventing structure the annotator did not confirm.
    """
    import pypdfium2 as pdfium

    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    rendered: dict[int, object] = {}
    counts: dict[str, int] = {}
    assets: list[_Asset] = []

    for idx, r in enumerate(regions):
        if r.type not in ASSET_TYPES:
            continue
        atype = ASSET_TYPES[r.type]
        counts[atype] = counts.get(atype, 0) + 1
        number = counts[atype]
        asset_id = f"{atype}_{number}"

        if r.page not in rendered:
            rendered[r.page] = doc[r.page].render(scale=OCR_SCALE).to_pil()
        page_img = rendered[r.page]
        x0, y0, x1, y1 = r.bbox
        crop = page_img.crop((max(0, int(x0)), max(0, int(y0)),
                              int(x1), int(y1)))
        image = assets_dir / f"{asset_id}.png"
        crop.save(image)

        section = headings_at.get(idx, "")
        caption = _caption_for(regions, idx) or (
            f"{section} — {atype.capitalize()} {number}" if section
            else f"{atype.capitalize()} {number}")
        assets.append(_Asset(
            asset_id=asset_id, type=atype, number=number, caption=caption,
            page=r.page + 1, image=str(image), bbox=list(r.bbox),
            sections=[section] if section else [],
            table_markdown=html_table_to_markdown(r.html),
            is_icon=idx in (icons or set())))
    return assets


def _caption_for(regions: list[Region], idx: int) -> str:
    """A caption region touching this asset — the one immediately before or
    after it in reading order, on the same page."""
    for j in (idx - 1, idx + 1):
        if 0 <= j < len(regions):
            r = regions[j]
            if r.type in CAPTION_TYPES and r.page == regions[idx].page and r.text:
                return r.text
    return ""


# ── markdown + pins ──────────────────────────────────────────────────────────

def build_markdown(regions: list[Region], assets: list[_Asset], stem: str,
                   asset_url_base: str,
                   golden: dict[int, list[str]] | None = None) -> str:
    """Regions → knowledge_base markdown with a pin under every heading.

    The pin emitters are imported from _massage so the two ingest paths cannot
    drift in pin FORMAT.  What differs is the join: betteringest_pdf has to
    match provenance back to headings by normalised title, because its markdown
    comes from a separate reconstruction.  Here we emit the markdown ourselves,
    so every heading's provenance is carried on the line index — exact, and
    with no way for a rewritten heading to lose its page and bbox.
    """
    from pageindex import _parse_headings
    from modules.ingest._massage import (_asset_pin, _caption_heading,
                                         _section_pin)

    def with_golden(pin_lines: list[str], prov) -> list[str]:
        """Append `golden: ...` to a pin whose chunk covers a labelled page.

        Inserted rather than passed to the pin emitters so the two ingest
        paths keep writing byte-identical pins for everything else — this is
        an eval annotation, not a change to the pin format.
        """
        if not golden or prov is None:
            return pin_lines
        pages = set()
        if isinstance(prov, dict):
            if prov.get("page"):
                pages.add(int(prov["page"]))
            pages.update(int(p) for p in (prov.get("regions") or {}))
        else:                                   # an _Asset
            pages.add(int(prov.page))
        labels = sorted({l for p in pages for l in golden.get(p, [])})
        if not labels:
            return pin_lines
        out = list(pin_lines)
        out.insert(out.index("```", 1), f"golden: {', '.join(labels)}")
        return out

    assets_by_index = {}
    ai = iter(assets)
    for idx, r in enumerate(regions):
        if r.type in ASSET_TYPES:
            assets_by_index[idx] = next(ai)
    caption_used = {j for idx in assets_by_index
                    for j in (idx - 1, idx + 1)
                    if 0 <= j < len(regions)
                    and regions[j].type in CAPTION_TYPES
                    and regions[j].page == regions[idx].page}

    lines: list[str] = []
    # line index of a heading → what its pin should say
    pin_for: dict[int, tuple] = {}
    current_level = 1
    prov: dict | None = None
    pending: list[_Asset] = []              # this section's assets, not yet emitted

    def emit_asset(a: _Asset, level: int) -> None:
        pin_for[len(lines)] = ("asset", a)
        lines.extend([f"{'#' * level} {_caption_heading(a.caption)}", ""])
        if a.table_markdown:
            # Extended line-by-line, never as one multi-line string: pins are
            # placed by LINE INDEX, so a blob that spans lines would silently
            # shift every heading after it out of alignment.
            lines.extend(a.table_markdown.split("\n") + [""])
        else:
            lines.extend([f"![{a.type} {a.number}]"
                          f"({asset_url_base}/{Path(a.image).name})", ""])
        body = " ".join(x for x in (a.caption.strip(),
                                    a.description.strip()) if x)
        if body:
            lines.extend(body.split("\n") + [""])

    def emit_icon_group(icons: list[_Asset], level: int) -> None:
        """All of this section's decoration as ONE leaf.

        Kept in the document rather than dropped — the crops and their page
        locations stay inspectable — but not given a heading each, because
        thirty-nine captionless nodes would be a third of the index saying
        nothing.  The body states plainly what they are and where, so the
        leaf's summary has something true to work from instead of a blank.
        """
        by_page: dict[int, int] = {}
        regions_box: dict[int, list[float]] = {}
        for a in icons:
            by_page[a.page] = by_page.get(a.page, 0) + 1
            x0, y0, x1, y1 = a.bbox
            cur = regions_box.get(a.page)
            regions_box[a.page] = ([x0, y0, x1, y1] if cur is None else
                                   [min(cur[0], x0), min(cur[1], y0),
                                    max(cur[2], x1), max(cur[3], y1)])
        pin_for[len(lines)] = ("section", {
            "page": icons[0].page, "bbox": list(icons[0].bbox),
            "regions": regions_box})
        lines.extend([f"{'#' * level} {ICON_GROUP_HEADING}", ""])
        where = ", ".join(f"{n} on page {pg}" for pg, n in sorted(by_page.items()))
        lines.extend([
            f"{len(icons)} small graphics of near-identical size ({where}) that "
            f"layout detection reported as figures. None carries a caption, so "
            f"they are most likely icons or fragments of one infographic rather "
            f"than figures in their own right. They are not described.", ""])
        for a in icons:
            lines.extend([f"![{a.type} {a.number}]"
                          f"({asset_url_base}/{Path(a.image).name})", ""])

    def flush_assets() -> None:
        """Close the open section by emitting its figures and tables.

        Assets go under ONE grouping subsection rather than straight into the
        section, because a section's summary is rolled up from its children's
        summaries: a section with fifteen figure leaves and one prose leaf
        gets summarised as being about figures, and the prose — the part that
        holds the facts — is outvoted fifteen to one.  Measured on the BPPL
        document: assets outnumbered prose in 12 of 18 sections, and the
        Executive summary was pruned for a query its own first sentence
        answers.  Grouping makes those children [prose, assets], so the
        roll-up weighs them evenly.

        One asset cannot outvote anything, so it is left in place — an extra
        heading level there would be noise.
        """
        nonlocal pending
        if not pending:
            return
        level = min(current_level + 1, MAX_LEVEL)
        icons = [a for a in pending if a.is_icon]
        real = [a for a in pending if not a.is_icon]
        pending = []
        if icons:
            emit_icon_group(icons, level)
        if not real:
            return
        if len(real) == 1:
            emit_asset(real[0], level)
            return
        pending = real
        kinds = {a.type for a in pending}
        label = ("Figures and tables" if len(kinds) > 1 else
                 "Tables" if "table" in kinds else "Figures")
        group = {"page": pending[0].page, "bbox": list(pending[0].bbox),
                 "regions": {}}
        for a in pending:                   # the group spans all its assets
            x0, y0, x1, y1 = a.bbox
            cur = group["regions"].get(a.page)
            group["regions"][a.page] = ([x0, y0, x1, y1] if cur is None else
                                        [min(cur[0], x0), min(cur[1], y0),
                                         max(cur[2], x1), max(cur[3], y1)])
        pin_for[len(lines)] = ("section", group)
        lines.extend([f"{'#' * level} {label}", ""])
        for a in pending:
            emit_asset(a, min(level + 1, MAX_LEVEL))
        pending = []

    for idx, r in enumerate(regions):
        if idx in caption_used and idx not in assets_by_index:
            continue                        # consumed as an asset-leaf heading
        if r.level is not None:
            if not r.text:
                continue
            flush_assets()
            current_level = r.level or 2
            prov = {"page": r.page + 1, "bbox": list(r.bbox), "regions": {}}
            pin_for[len(lines)] = ("section", prov)
            lines += [f"{'#' * current_level} {r.text}", ""]
        elif idx in assets_by_index:
            pending.append(assets_by_index[idx])
        elif r.text:
            # Body text also widens the open section's per-page region — one
            # union box per page, not one per paragraph.
            if prov is not None:
                page = r.page + 1
                x0, y0, x1, y1 = r.bbox
                cur = prov["regions"].get(page)
                prov["regions"][page] = ([x0, y0, x1, y1] if cur is None else
                                         [min(cur[0], x0), min(cur[1], y0),
                                          max(cur[2], x1), max(cur[3], y1)])
            lines += r.text.split("\n") + [""]

    flush_assets()                          # the last section's assets
    text = "\n".join(lines)
    # Node ids exactly as the indexer will assign them — the pin id IS the join
    # key between a retrieved chunk, the doc viewer and the PDF.
    pins_at: dict[int, list[str]] = {}
    for line_idx, _level, _title, node_id in _parse_headings(text):
        entry = pin_for.get(line_idx)
        if entry is None:
            continue
        kind, payload = entry
        if kind == "asset":
            pins_at[line_idx] = with_golden(_asset_pin(
                node_id, stem, payload,
                f"{asset_url_base}/{Path(payload.image).name}", OCR_SCALE),
                payload)
        else:
            pins_at[line_idx] = with_golden(
                _section_pin(node_id, stem, payload, OCR_SCALE), payload)

    out = text.split("\n")
    for line_idx in sorted(pins_at, reverse=True):
        insert_at = line_idx + 1
        if insert_at < len(out) and out[insert_at].strip() == "":
            insert_at += 1
        out[insert_at:insert_at] = pins_at[line_idx]
    return "\n".join(out)


# ── the ingest stage ─────────────────────────────────────────────────────────

def _clean_title(text: str, max_len: int = 160) -> str:
    """A document title fit for the library list: single line, trimmed."""
    text = " ".join(text.split()).strip()
    return text if len(text) <= max_len else text[:max_len - 1].rstrip() + "…"


def _stem_for(state: dict, session: Path) -> str:
    """Document id from the PDF the session was cut from, not from the session
    folder — every session holds a file literally called source.pdf, so a
    path-derived id would collide for every document in the corpus."""
    name = (state.get("document", {}).get("filename")
            or f"{session.name}.pdf")
    stem = Path(name).stem
    stem = re.sub(r"_origin$", "", stem)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or session.name


def _kept_pdf_path(session: Path, root: Path | None, stem: str) -> Path:
    """Where this document's PDF lives inside the library.

    The session's PARENT folders are mirrored under knowledge_base/sources/ so
    the manifest derives the same folder tags it would for a plain PDF corpus.
    The session folder itself is dropped — its name is a uuid, not a label.
    """
    parts: tuple[str, ...] = ()
    if root is not None:
        try:
            parts = session.resolve().relative_to(root.resolve()).parts[:-1]
        except (ValueError, OSError):
            parts = ()
    return PDF_DIR.joinpath(*parts, f"{stem}.pdf")


def run(source_dir: str | None = None, progress=None) -> dict:
    """Ingest every annotation session under the source folder."""
    src = Path(source_dir or get_source_dir() or "")
    if not src or not src.is_dir():
        raise FileNotFoundError(
            f"MinerU source folder not set or missing: '{src}'. Pick a folder "
            "of annotation sessions in the app (or pass source_dir).")
    if source_dir:
        set_source_dir(str(src))
    sessions = discover_sessions(src)
    if not sessions:
        raise FileNotFoundError(
            f"No annotation sessions (folders with {STATE_NAME}) under {src}/")
    return _ingest(sessions, src, progress)


def run_paths(paths: list[str], progress=None) -> dict:
    """Additive ingest of explicit session folders, on top of whatever is
    already in knowledge_base/. Powers the Library's '＋ Add' affordance."""
    sessions: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(f"Not an annotation session folder: {p}")
        found = discover_sessions(p)
        if not found:
            raise FileNotFoundError(f"No {STATE_NAME} under {p}")
        sessions.extend(found)
    return _ingest(sessions, None, progress)


def _ingest(sessions: list[Path], root: Path | None, progress=None) -> dict:
    from modules.ingest._manifest import Manifest

    KB_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(KB_DIR)
    sources = _load_sources()
    warnings: list[str] = []
    done: list[str] = []

    def report(phase: str, doc: str, i: int, message: str = "") -> None:
        if progress:
            progress({"phase": phase, "doc": doc, "done": i,
                      "total": len(sessions), "message": message})

    for i, session in enumerate(sessions):
        state_path = session / STATE_NAME
        pdf_path = session / PDF_NAME
        state = json.loads(state_path.read_text(encoding="utf-8"))
        stem = _stem_for(state, session)
        report("parse", stem, i, f"Reading {session.name}…")

        if not pdf_path.is_file():
            # Flagged, never skipped silently: without the PDF there is no body
            # text and no crops, and the pins would point at nothing.
            warnings.append(f"{stem}: no {PDF_NAME} in {session} — skipped")
            print(f"  [warn] {warnings[-1]}", file=sys.stderr)
            continue

        regions = load_regions(state)
        if not regions:
            warnings.append(f"{stem}: no content regions — skipped")
            continue

        report("text", stem, i, f"Reading text layer for {stem}…")
        empty = fill_text_from_pdf(regions, pdf_path)
        if empty:
            warnings.append(
                f"{stem}: {empty} region(s) had no text layer (scanned pages?) "
                f"— ingested without their text")

        # A section title for each asset, for captions the session did not give.
        headings_at: dict[int, str] = {}
        current = ""
        for idx, r in enumerate(regions):
            if r.level is not None and r.text:
                current = r.text
            headings_at[idx] = current

        report("assets", stem, i, f"Cropping assets of {stem}…")
        work_dir = OUT_DIR / stem
        icons = flag_icon_clusters(regions)
        assets = _crop_assets(regions, pdf_path, work_dir, headings_at, icons)
        if icons:
            warnings.append(
                f"{stem}: {len(icons)} asset(s) grouped as "
                f"'{ICON_GROUP_HEADING}' — same-size uncaptioned runs, "
                f"not described")

        if assets:
            report("captions", stem, i,
                   f"Captioning {len(assets)} asset(s) of {stem}…")

            def _caption_progress(done: int, total: int,
                                  _stem: str = stem, _i: int = i) -> None:
                # A 73-asset document spends ~15s per crop in the vision model.
                # Without this the app shows one "Captioning…" line and then
                # nothing for twenty minutes, which reads as a hang.
                report("captions", _stem, _i,
                       f"Captioning {_stem}: asset {done + 1}/{total}…")

            _describe(assets, warnings, stem, _caption_progress)

        report("massage", stem, i, f"Writing {stem}.md with provenance pins…")
        golden = load_golden(stem)
        markdown = build_markdown(regions, assets, stem,
                                  asset_url_base=f"/assets/{stem}",
                                  golden=golden)
        if golden:
            report("massage", stem, i,
                   f"Stamping golden labels on {len(golden)} page(s) of {stem}…")
        if assets:
            asset_dir = KB_DIR / "assets" / stem
            asset_dir.mkdir(parents=True, exist_ok=True)
            for a in assets:
                shutil.copy2(a.image, asset_dir / Path(a.image).name)
        (KB_DIR / f"{stem}.md").write_text(markdown, encoding="utf-8")

        # The PDF moves into the library: pins outlive the session folder.
        kept_pdf = _kept_pdf_path(session, root, stem)
        kept_pdf.parent.mkdir(parents=True, exist_ok=True)
        if not kept_pdf.exists() or kept_pdf.stat().st_size != pdf_path.stat().st_size:
            shutil.copy2(pdf_path, kept_pdf)

        sources[stem] = {
            "pdf": str(kept_pdf.resolve()),
            "ocr_scale": OCR_SCALE,
            "module": MODULE_INFO["name"],
            "assets": [a.to_dict() for a in assets],
        }
        SOURCES_MANIFEST.write_text(
            json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
        # Registered against the KEPT copy, not the session's source.pdf: the
        # manifest takes a document's title from its filename stem and its tags
        # from the folders above it, and every session holds a file called
        # source.pdf inside a uuid-named folder — which would title every
        # annotated document "source" and tag it with a uuid.
        entry = manifest.register(
            source_path=kept_pdf, root=PDF_DIR,
            ingest_module=MODULE_INFO["name"], doc_id=stem,
            extra={"pdf": str(kept_pdf.resolve()),
                   "ocr_scale": OCR_SCALE,
                   "session": str(session.resolve()),
                   "assets": [a.to_dict() for a in assets]})
        # The manifest titles a document by its FILENAME, which for these is
        # whatever the extraction pipeline named the PDF — often a uuid
        # ("ddb35d27_41b50400-a607-11f1-8b82-5f5342bdc0ca"), unreadable in the
        # library. The document states its own title in a doc_title region, so
        # use that and keep the filename only as a fallback.
        title = next((r.text for r in regions
                      if r.level is not None and r.text
                      and r.type == "doc_title"), "")
        if title and title != entry.title:
            entry.title = _clean_title(title)
            manifest.save()
        done.append(stem)
        print(f"  {session.name} → {KB_DIR / (stem + '.md')} "
              f"({len(assets)} assets, {len(regions)} regions)")
        report("done-doc", stem, i + 1)

    print(f"\nIngested {len(done)} session(s) into {KB_DIR}/")
    return {"docs": done, "warnings": warnings}


def _describe(assets: list[_Asset], warnings: list[str], stem: str,
              progress_cb=None) -> None:
    """Retrieval-oriented descriptions for the crops, via the shared captioning
    backends (local Ollama by default, write-through cached by crop sha).

    Flagged, not silently downgraded: without descriptions the asset leaves
    still carry their captions and tables, but the user is told.
    """
    try:
        from modules.ingest._betteringest import BetterIngest
        from modules.ingest._captioning import CaptioningUnavailable, caption_assets
    except ImportError as exc:
        warnings.append(f"{stem}: asset descriptions skipped — {exc}")
        return

    class _Doc:                    # caption_assets only needs `.assets`
        def __init__(self, assets): self.assets = assets

    # Icons are not described. A vision model handed a pictogram invents
    # meaning for it ("a cluster of discrete data points representing a
    # Critical group"), and that invention would be indexed as fact — so the
    # honest output is no description, and the vision calls are saved.
    describable = [a for a in assets if not a.is_icon]
    if not describable:
        return
    try:
        caption_assets(BetterIngest(out_dir=OUT_DIR), _Doc(describable),
                       cache_dir=OUT_DIR, progress_cb=progress_cb)
    except CaptioningUnavailable as exc:
        warnings.append(f"{stem}: asset descriptions skipped — {exc}")
        print(f"  [warn] {warnings[-1]}", file=sys.stderr)
