"""
ingest/reconstruct.py — turn OCR-detected headings into a tree by their numbering.

The rendered numbering is the cleanest level signal (far more reliable than font
size, which inverts on IEEE small-caps).  It is sitting in the OCR'd heading text:

  dotted arabic  "1", "1.1", "1.2.3"   → level = number of components   (article)
  roman          "I.", "II."           → level 1                        (IEEE section)
  letter         "A.", "B.", "D."      → level 2                        (IEEE subsection)
  digit + paren  "1)", "2)"            → level 3                        (IEEE subsubsection)

The hard case: single letters like `I`, `V`, `C`, `D` are valid roman numerals
*and* subsection letters, and IEEEtran renders sections in Title Case (so the
caps signal is unreliable).  We resolve it by **sequence**: a token is a section
if it continues the roman section run (I, II, III, …), and a subsection if it
continues the letter run (A, B, C, D, …).  Headings with no numbering fall back
to a flat list (correct for unnumbered documents, which are typically flat).
"""

from __future__ import annotations

import re
from pathlib import Path

from .tree import Node, normalise_title, normalize_to_leaves
from .ocr import Block

_SECTION_TITLE = "paragraph_title"

_PAREN_RE = re.compile(r"^\s*\d+\)\s")
_DOTTED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?(?=\s|$|[A-Za-z])")
_ALPHA_RE = re.compile(r"^\s*([A-Za-z]+)\.\s")
_ROMAN_VALUE = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _roman_to_int(s: str) -> int | None:
    s = s.lower()
    total = prev = 0
    for ch in reversed(s):
        if ch not in _ROMAN_VALUE:
            return None
        v = _ROMAN_VALUE[ch]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def _classify(text: str):
    """Return ('paren',) | ('dotted', level) | ('alpha', token) | None."""
    if _PAREN_RE.match(text):
        return ("paren",)
    m = _DOTTED_RE.match(text)
    if m:
        return ("dotted", m.group(1).count(".") + 1)
    m = _ALPHA_RE.match(text)
    if m:
        return ("alpha", m.group(1))
    return None


def assign_levels(headings: list[str], sizes: list[float | None] | None = None) -> list[int]:
    """Levels for headings in reading order.

    Numbered documents → numbering (reliable).  Unnumbered documents → font-size
    ranking (largest size = level 1), if sizes are available; else flat.  Font
    size is unreliable in general (it inverts on IEEE small-caps), but IEEE docs
    are numbered and so take the numbering path — the font fallback only fires on
    unnumbered docs, where size decreases with depth.
    """
    n_numbered = sum(1 for h in headings if _classify(h) is not None)
    if 2 * n_numbered >= len(headings) and headings:
        return _numbering_levels(headings)

    if sizes and 2 * sum(1 for s in sizes if s) >= len(headings):
        ranked = {s: i + 1 for i, s in enumerate(sorted({s for s in sizes if s}, reverse=True))}
        return [ranked.get(s, 1) for s in sizes]

    return [1] * len(headings)                              # no signal → flat


def _numbering_levels(headings: list[str]) -> list[int]:
    """Levels from rendered numbering, resolving roman/letter by sequence."""
    out: list[int] = []
    next_sec = 1   # expected roman value of the next section
    next_sub = 0   # expected letter index (A=0) of the next subsection

    for h in headings:
        c = _classify(h)
        if c is None:
            out.append(1)                                  # unnumbered heading in a numbered doc
            continue
        if c[0] == "paren":
            out.append(3)
            continue
        if c[0] == "dotted":
            out.append(c[1])
            continue

        token = c[1]                                       # 'alpha'
        rv = _roman_to_int(token)
        li = (ord(token.upper()) - ord("A")) if len(token) == 1 else None

        if len(token) > 1 and rv is not None:              # "II", "IV" → section
            out.append(1); next_sec = rv + 1; next_sub = 0
        elif rv is not None and rv == next_sec:            # single roman continuing sections
            out.append(1); next_sec = rv + 1; next_sub = 0
        elif li is not None and li == next_sub:            # letter continuing subsections
            out.append(2); next_sub = li + 1
        elif li is not None:                               # a letter, off-sequence → subsection
            out.append(2); next_sub = li + 1
        elif rv is not None:                               # roman, off-sequence → section
            out.append(1); next_sec = rv + 1; next_sub = 0
        else:
            out.append(1)
    return out


def reading_order(blocks: list[Block]) -> list[Block]:
    """Sort blocks into reading order, column-aware (for 2-column layouts).

    Per page: a block is in the right column if its LEFT edge starts past the page
    midline (1-column blocks all start at the left margin, so they stay column 0).
    Within a column, top-to-bottom; left column before right.
    """
    by_page: dict[int, list[Block]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)

    ordered: list[Block] = []
    for page in sorted(by_page):
        pb = by_page[page]
        left = min(b.bbox[0] for b in pb)
        right = max(b.bbox[2] for b in pb)
        mid = (left + right) / 2
        ordered += sorted(pb, key=lambda b: (1 if b.bbox[0] > mid else 0, b.bbox[1]))
    return ordered


def font_sizes_for_headings(
    head_blocks: list[Block], pdf_path, ocr_scale: float = 2.0,
) -> list[float | None]:
    """Dominant text-layer font size per heading block (None if no text layer)."""
    from collections import Counter

    import pypdfium2 as pdfium  # type: ignore[import-untyped]
    import pypdfium2.raw as R   # type: ignore[import-untyped]

    doc = pdfium.PdfDocument(str(pdf_path))
    by_page: dict[int, list] = {}
    for i, b in enumerate(head_blocks):
        by_page.setdefault(b.page, []).append((i, b))

    sizes: list[float | None] = [None] * len(head_blocks)
    s = 1.0 / ocr_scale
    for page_idx, entries in by_page.items():
        page = doc[page_idx]
        ph = page.get_height()
        tp = page.get_textpage()
        chars = [(tp.get_charbox(i), R.FPDFText_GetFontSize(tp, i))
                 for i in range(tp.count_chars())]
        for idx, b in entries:
            x0, y0, x1, y1 = b.bbox
            bl, bb, br, bt = x0 * s, ph - y1 * s, x1 * s, ph - y0 * s
            hits = [round(fs * 2) / 2 for (cl, cb, cr, ct), fs in chars
                    if fs >= 3 and cl < br + 4 and cr > bl - 4 and cb < bt + 4 and ct > bb - 4]
            if hits:
                sizes[idx] = Counter(hits).most_common(1)[0][0]
    return sizes


def build_ocr_tree(blocks: list[Block], doc_title: str = "", pdf_path=None,
                   ocr_scale: float = 2.0) -> Node:
    """Build a heading tree from OCR title blocks.

    Levelled by rendered numbering; for unnumbered docs, by text-layer font size
    when `pdf_path` is given (born-digital).  Reading-order aware (2-column safe).
    """
    head_blocks = [b for b in reading_order(blocks) if b.label == _SECTION_TITLE]
    headings = [b.text for b in head_blocks]
    sizes = font_sizes_for_headings(head_blocks, pdf_path, ocr_scale) if pdf_path else None
    levels = assign_levels(headings, sizes)

    root = Node(title=doc_title)
    stack: list[tuple[int, Node]] = [(0, root)]
    for title, level in zip(headings, levels):
        node = Node(title=title)
        while stack and stack[-1][0] >= level:
            stack.pop()
        (stack[-1] if stack else (0, root))[1].children.append(node)
        stack.append((level, node))
    return root


# subfigure subcaption: "(a) …", "(b) …" — a panel label, not a top-level caption
_SUBCAPTION = re.compile(r"^\s*\(?[a-zA-Z]\)\s")


def recovered_assets(blocks: list[Block]) -> list[Node]:
    """Assets detected by OCR: each `figure_title` block is a caption; its type is
    inferred from the rendered label ("Table 1:" → table, else figure).  Subfigure
    panel labels ("(a) …") are skipped — they are not top-level float captions."""
    out: list[Node] = []
    for b in reading_order(blocks):
        if b.label == "figure_title" and b.text.strip() and not _SUBCAPTION.match(b.text):
            c = b.text.strip().lower()
            atype = "table" if c.startswith(("table", "tab.")) else "figure"
            out.append(Node(title=b.text.strip(), kind="asset", asset_type=atype))
    return out


def block_text(block: Block, textpage, page_height: float, ocr_scale: float = 2.0) -> str:
    """Text for one block from the PDF text layer (born-digital → clean, ~1ms, no
    OCR).  Returns "" when the layer is empty (scanned/garbled) — the seam where a
    real OCR fallback (RapidOCR/ONNX) plugs in later."""
    s = 1.0 / ocr_scale
    x0, y0, x1, y1 = block.bbox
    txt = textpage.get_text_bounded(
        left=x0 * s, bottom=page_height - y1 * s, right=x1 * s, top=page_height - y0 * s)
    txt = txt.replace("\x02", "").replace("\xad", "")     # discretionary/soft hyphens
    txt = " ".join(txt.split())
    return re.sub(r"(\w)-\s(\w)", r"\1\2", txt)           # join line-break hyphenation


# "Figure 3", "Fig. 3", "Table 1", "Tab 2" → (type, number) mention in prose
_REF_MENTION = re.compile(r"\b(fig(?:ure)?|tab(?:le)?)s?\.?\s*(\d+)", re.I)


def recovered_reachability(blocks: list[Block], pdf_path,
                           ocr_scale: float = 2.0) -> set[tuple[str, str, int]]:
    """{(normalised section heading, asset_type, number)} — a section is linked to
    an asset iff its body text mentions it ("Figure 3").  Section = the most-recent
    heading in reading order; body text via the text layer (Phase-3b on path A)."""
    import pypdfium2 as pdfium  # type: ignore[import-untyped]
    from .tree import normalise_title

    doc = pdfium.PdfDocument(str(pdf_path))
    tp: dict[int, tuple] = {}

    def page(p: int):
        if p not in tp:
            pg = doc[p]
            tp[p] = (pg.get_textpage(), pg.get_height())
        return tp[p]

    out: set[tuple[str, str, int]] = set()
    cur = ""
    for b in reading_order(blocks):
        if b.label == "paragraph_title":
            cur = normalise_title(b.text)
        elif b.label == "text" and cur:
            page_tp, ph = page(b.page)
            for m in _REF_MENTION.finditer(block_text(b, page_tp, ph, ocr_scale)):
                atype = "figure" if m.group(1).lower().startswith("fig") else "table"
                out.add((cur, atype, int(m.group(2))))
    return out


def build_content_tree(blocks: list[Block], doc_title: str = "", pdf_path=None,
                       ocr_scale: float = 2.0,
                       levels: list[int] | None = None) -> Node:
    """Recovered content-at-leaves tree: heading hierarchy (numbering/font levels)
    with body text (text layer) anchored under the deepest current heading, then
    `normalize_to_leaves` so content lives only in leaves.  Mirrors how the LaTeX
    GT (`parse_doc`) is built, so the two are comparable leaf-for-leaf.

    `levels` overrides the internal level assigner with one entry per detected
    heading (reading order) — the seam that lets the escalation ladder's stronger
    hierarchy carry the content partition (BetterIngest uses this).  A None
    entry DEMOTES that heading to body text (pseudo-heading the validated
    outline disowns: 'Abstract', 'Contributions.', running heads)."""
    import pypdfium2 as pdfium  # type: ignore[import-untyped]

    ordered = reading_order(blocks)
    head_blocks = [b for b in ordered if b.label == _SECTION_TITLE]
    if levels is None:
        sizes = font_sizes_for_headings(head_blocks, pdf_path, ocr_scale) if pdf_path else None
        levels = assign_levels([b.text for b in head_blocks], sizes)
    elif len(levels) != len(head_blocks):
        raise ValueError(f"levels: expected {len(head_blocks)} entries, "
                         f"got {len(levels)}")
    level_of = {id(b): lv for b, lv in zip(head_blocks, levels)}
    head_blocks = [b for b in head_blocks if level_of[id(b)] is not None]

    # Page-0 title/author band: column-aware order can sort a right-column author
    # block *after* a left-column heading, bleeding it into that section.  Anything
    # above the topmost heading on page 0 is frontmatter — drop it.
    p0_heads = [b.bbox[1] for b in head_blocks if b.page == 0]
    frontmatter_y = min(p0_heads) if p0_heads else -1.0

    doc = pdfium.PdfDocument(str(pdf_path)) if pdf_path else None
    tp: dict[int, tuple] = {}

    def text_of(b: Block) -> str:
        if doc is None:
            return ""
        if b.page not in tp:
            pg = doc[b.page]
            tp[b.page] = (pg.get_textpage(), pg.get_height())
        page_tp, ph = tp[b.page]
        return block_text(b, page_tp, ph, ocr_scale)

    root = Node(title=doc_title)
    stack: list[tuple[int, Node]] = [(0, root)]
    for b in ordered:
        if b.label == _SECTION_TITLE and level_of.get(id(b)) is not None:
            node = Node(title=b.text)
            lv = level_of[id(b)]
            while stack and stack[-1][0] >= lv:
                stack.pop()
            (stack[-1] if stack else (0, root))[1].children.append(node)
            stack.append((lv, node))
        elif b.label == "text" or (b.label == _SECTION_TITLE
                                   and id(b) in level_of):
            cur = stack[-1][1]
            # Pre-first-heading text is frontmatter → drop, mirroring GT — but
            # only in a *headed* document.  A document with no headings at all
            # (memo, notice, single-asset sheet) is one leaf in GT (parse_doc
            # puts the whole body on the root), so its text anchors to the root
            # instead of being dropped.
            if cur is root and head_blocks:
                continue
            if b.page == 0 and b.bbox[1] < frontmatter_y:   # title/author band
                continue
            txt = text_of(b)
            if txt:
                cur.content = (cur.content + " " + txt).strip()
    normalize_to_leaves(root)
    _prune_empty(root)
    return root


def _prune_empty(node: Node) -> None:
    """Drop heading nodes that ended up with no content and no children (e.g. an
    'Abstract' heading whose body is dropped frontmatter)."""
    for c in node.children:
        _prune_empty(c)
    node.children = [c for c in node.children
                     if c.kind == "content" or c.children or c.content.strip()]


_ASSET_CONTENT = {"image", "chart", "table"}


def save_asset_crops(blocks: list[Block], pdf_path, out_dir,
                     ocr_scale: float = 2.0) -> list[dict]:
    """Crop each detected asset (figure/table) to a PNG and return a manifest.

    Each `figure_title` caption is paired with the nearest content block
    (`image`/`chart`/`table`) in its column (horizontal overlap, smallest vertical
    gap); the saved crop is the union of caption + content, so it's self-contained.
    If no content block is near (e.g. a text-rendered table), the caption region
    alone is saved.  bboxes are in render pixels at `ocr_scale`, so we re-render at
    the same scale and crop directly.
    """
    import pypdfium2 as pdfium  # type: ignore[import-untyped]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    captions = [b for b in reading_order(blocks)
                if b.label == "figure_title" and b.text.strip()
                and not _SUBCAPTION.match(b.text)]
    content = [b for b in blocks if b.label in _ASSET_CONTENT]

    doc = pdfium.PdfDocument(str(pdf_path))
    rendered: dict[int, object] = {}

    def page_img(p: int):
        if p not in rendered:
            rendered[p] = doc[p].render(scale=ocr_scale).to_pil()
        return rendered[p]

    manifest: list[dict] = []
    used: set[int] = set()
    count = {"figure": 0, "table": 0}
    for cap in captions:
        atype = "table" if cap.text.strip().lower().startswith(("table", "tab.")) else "figure"
        cx0, cy0, cx1, cy1 = cap.bbox
        ph = page_img(cap.page).height
        best, best_gap = None, ph * 0.30                # cap search to ~⅓ page
        for i, cb in enumerate(content):
            if i in used or cb.page != cap.page:
                continue
            bx0, by0, bx1, by1 = cb.bbox
            if min(cx1, bx1) - max(cx0, bx0) <= 0:      # require same-column overlap
                continue
            gap = max(by0 - cy1, cy0 - by1, 0.0)        # vertical gap (above or below)
            if gap < best_gap:
                best, best_gap = i, gap

        boxes = [cap.bbox]
        if best is not None:
            used.add(best)
            boxes.append(content[best].bbox)
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)

        count[atype] += 1
        fname = f"{atype}_{count[atype]}.png"
        page_img(cap.page).crop((int(x0), int(y0), int(x1), int(y1))).save(out_dir / fname)
        nm = re.search(r"\d+", cap.text)              # rendered number ("Figure 3" → 3)
        manifest.append({
            "type": atype, "caption": cap.text.strip(),
            "number": int(nm.group()) if nm else count[atype],
            "image": str(out_dir / fname), "page": cap.page,
            "has_content": best is not None,
            # astepsis addition: crop bbox (render px at ocr_scale) so pins can
            # locate the asset in the source PDF without re-detecting it.
            "bbox": [x0, y0, x1, y1],
        })
    return manifest


def to_markdown(root: Node, assets_at: dict[str, list[str]] | None = None) -> str:
    """Render the content-at-leaves tree as markdown.  Artificial leaves (title ==
    parent) emit their text with no heading; real headings emit `#`*depth.  Asset
    image links in `assets_at[normalised heading]` are placed under that leaf
    (each once); any left over go in a trailing section."""
    assets_at = assets_at or {}
    lines: list[str] = [f"# {root.title}\n"] if root.title else []
    placed: set[str] = set()

    # An unheaded document: normalize_to_leaves turned the root itself into the
    # single content leaf, so there are no children to walk — emit its body here.
    if root.kind == "content" and root.content:
        lines.append(root.content + "\n")

    def emit(node: Node, depth: int) -> None:
        for c in node.children:
            hashes = "#" * min(depth, 6)
            if c.kind != "content":
                lines.append(f"{hashes} {c.title}\n")
                emit(c, depth + 1)
                continue
            if normalise_title(c.title) != normalise_title(node.title):
                lines.append(f"{hashes} {c.title}\n")
            if c.content:
                lines.append(c.content + "\n")
            for img in assets_at.get(normalise_title(c.title), []):
                if img not in placed:
                    lines.append(img + "\n")
                    placed.add(img)

    emit(root, 2)                                     # doc title is H1, sections H2
    leftover = [img for imgs in assets_at.values() for img in imgs if img not in placed]
    if leftover:
        lines.append("## Figures and Tables\n")
        lines += [img + "\n" for img in dict.fromkeys(leftover)]
    return "\n".join(lines)


def ocr_doc_title(blocks: list[Block], default: str = "") -> str:
    for b in blocks:
        if b.label == "doc_title" and b.text.strip():
            return b.text.strip()
    return default
