"""
ingest/outline.py — ladder rung 0: the embedded PDF outline, validation-gated.

Design per docs/heuristic-robustness.md §3b: the gate is PER-ENTRY, never a
whole-outline accept/reject — service entries (O3) and run-in-deep entries
(O5) are pruned rather than sinking a faithful outline, and a flat outline
(O7) surrenders its levels but keeps its titles.  Verdicts:

  full        titles + levels trusted (high confidence)
  titles_only outline titles trusted, levels come from lower rungs
  reject      fall through to rung 1 entirely

Validation matches each entry against the OCR-detected title blocks (fuzzy,
after normalisation; page anchor must agree within a tolerance), which also
back-fills the bbox an outline lacks.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tree import Node, normalise_title

# Service/frontmatter entries pruned (O3) — deliberately minimal; "references"
# stays because the ground truth counts it as a section.
_SERVICE = {"abstract", "keywords", "contents", "table of contents",
            "list of figures", "list of tables", "acknowledgments",
            "acknowledgements", "cover"}


@dataclass
class OutlineEntry:
    title: str
    level: int          # 0-based as read from the PDF
    page: int           # destination page index (-1 if unresolvable)


def read_outline(pdf_path) -> list[OutlineEntry]:
    """Raw bookmark tree of a PDF, [] when absent (O1)."""
    import pypdfium2 as pdfium  # type: ignore[import-untyped]

    doc = pdfium.PdfDocument(str(pdf_path))
    out: list[OutlineEntry] = []
    for e in doc.get_toc():
        try:
            dest = e.get_dest()
            page = dest.get_index() if dest is not None else -1
        except Exception:
            page = -1
        title = (e.get_title() or "").strip()
        if title:
            out.append(OutlineEntry(title=title, level=e.level, page=page))
    return out


def _words(s: str) -> set[str]:
    return {w for w in s.split() if len(w) >= 2}


def _fuzzy_match(a: str, b: str) -> bool:
    """Normalised-title match: equality, or content-word overlap ≥ 0.7 —
    tolerant of encoding drift / dropped math (O4)."""
    na, nb = normalise_title(a), normalise_title(b)
    if na == nb and na:
        return True
    wa, wb = _words(na), _words(nb)
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= 0.7


@dataclass
class OutlineVerdict:
    verdict: str                       # full | titles_only | reject
    entries: list[OutlineEntry]        # pruned, unwrapped entries
    match_rate: float
    notes: list[str]


def validate_outline(entries: list[OutlineEntry], blocks,
                     page_tol: int = 1) -> OutlineVerdict:
    """Per-entry gate against detected title blocks (see module docstring)."""
    notes: list[str] = []
    if not entries:
        return OutlineVerdict("reject", [], 0.0, ["absent (O1)"])

    # O9 wrapper unwrap: a single level-0 entry with everything below it.
    top = [e for e in entries if e.level == 0]
    if len(top) == 1 and len(entries) > 1:
        entries = [OutlineEntry(e.title, e.level - 1, e.page)
                   for e in entries if e is not top[0]]
        notes.append("unwrapped single root (O9)")

    # O3 service prune (on the un-normalised lower-cased title).
    kept = [e for e in entries
            if normalise_title(e.title) not in _SERVICE]
    if len(kept) < len(entries):
        notes.append(f"pruned {len(entries) - len(kept)} service entries (O3)")
    entries = kept
    if not entries:
        return OutlineVerdict("reject", [], 0.0, notes + ["nothing after prune"])

    # O8 page-anchor garbage: entries titled like page markers.
    pageish = sum(1 for e in entries
                  if normalise_title(e.title).startswith("page "))
    if pageish >= max(3, len(entries) // 2):
        return OutlineVerdict("reject", [], 0.0, notes + ["page anchors (O8)"])

    # Per-entry match against detected title blocks (paragraph_title/doc_title
    # with recognised text).
    cands = [(b.text, b.page) for b in blocks
             if b.label in ("paragraph_title", "doc_title") and b.text.strip()]
    matched: list[OutlineEntry] = []
    for e in entries:
        hit = any(_fuzzy_match(e.title, t)
                  and (e.page < 0 or p < 0 or abs(e.page - p) <= page_tol)
                  for t, p in cands)
        if hit:
            matched.append(e)
    match_rate = len(matched) / len(entries)
    pruned = len(entries) - len(matched)
    if pruned:
        notes.append(f"pruned {pruned} unmatched entries (O5)")

    if match_rate < 0.5 or len(matched) < 2:
        return OutlineVerdict("reject", [], match_rate,
                              notes + [f"match rate {match_rate:.2f} < 0.5"])

    # O10 stale check: destination pages must be monotone non-decreasing.
    pages = [e.page for e in matched if e.page >= 0]
    if any(b < a for a, b in zip(pages, pages[1:])):
        return OutlineVerdict("reject", [], match_rate,
                              notes + ["non-monotone destinations (O10)"])

    # O7 flat outline: titles are valuable, levels are not.
    if len({e.level for e in matched}) == 1 and len(matched) > 3:
        return OutlineVerdict("titles_only", matched, match_rate,
                              notes + ["flat levels (O7)"])

    return OutlineVerdict("full", matched, match_rate, notes)


def outline_tree(entries: list[OutlineEntry], doc_title: str) -> Node:
    """Tree from validated outline entries (levels are 0-based)."""
    root = Node(title=doc_title)
    stack: list[tuple[int, Node]] = [(-1, root)]
    for e in entries:
        node = Node(title=e.title)
        while stack and stack[-1][0] >= e.level:
            stack.pop()
        (stack[-1] if stack else (-1, root))[1].children.append(node)
        stack.append((e.level, node))
    return root
