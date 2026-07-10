"""
ingest/levels.py — the escalation ladder's deterministic leveling core.

Replaces the closed-grammar heuristic (ingest/reconstruct.py:assign_levels)
with the redesign specified in docs/heuristic-robustness.md §4--§5:

  rung 1'  marker-schema INDUCTION (D1+D2+D4): an open signature vocabulary,
           and the document's own signature→level mapping inferred from
           counter-restart evidence — IEEE's roman→1/letter→2/paren→3 falls
           out as a special case instead of being hardcoded.  Per-heading
           routing replaces the 50% document-global gate (F3), so mixed-mode
           documents keep the levels their numbered headings carry.
  rung 2   TYPOGRAPHY clusters (D3): (font size, font weight) signatures,
           calibrated against numbered headings where any exist; unnumbered
           documents get cluster-rank levels; conflicts degrade to flat.
  rung 3   CONFIDENCE (D5): every heading carries (level, confidence, signal)
           so downstream consumers can route around guesses.

Every heading gets a `Leveled` record; `induce_levels` is pure (unit-testable
on plain strings), typography is injected as an optional parallel list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Marker signatures — the OPEN vocabulary (F1 fix)
# ---------------------------------------------------------------------------

_ROMAN_VALUE = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def roman_to_int(s: str) -> int | None:
    s = s.lower()
    total = prev = 0
    for ch in reversed(s):
        if ch not in _ROMAN_VALUE:
            return None
        v = _ROMAN_VALUE[ch]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total if total > 0 else None


_DOTTED = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?(?=\s|$)")
_LETTER_DOTTED = re.compile(r"^\s*([A-Z])\.(\d+(?:\.\d+)*)[.)]?(?=\s|$)")   # A.1 / B.2.3
_ALPHA_DOT = re.compile(r"^\s*([A-Za-z]+)[.)]\s")                            # A. / IV. / a)
_PAREN_DIGIT = re.compile(r"^\s*\(?(\d+)\)\s")
_PAREN_ALPHA = re.compile(r"^\s*\(([a-z])\)\s|^\s*([a-z])\)\s")
_PAREN_ROMAN = re.compile(r"^\s*\(([ivx]+)\)\s")
_WORD_PREFIX = re.compile(
    r"^\s*(Section|Chapter|Part|Annex|Appendix|Step|Question|Q)\.?\s*"
    r"(\d+(?:\.\d+)*|[A-Z]\b|[IVXLCDM]+\b)[.:)]?\s", re.I)


@dataclass(frozen=True)
class Marker:
    kind: str          # dotted | letterdotted | alpha | paren_digit | paren_alpha
                       # | paren_roman | word:<prefix>
    ordinal: int       # position in the series (1-based) for restart detection
    depth_hint: int    # extra depth within the family (dotted components - 1)
    raw: str           # the matched marker text
    ambiguous: bool = False   # single-token roman/letter ambiguity ("C.", "I.")


def classify_marker(text: str) -> Marker | None:
    """Open-vocabulary marker signature for one heading, or None."""
    m = _WORD_PREFIX.match(text)
    if m:
        word = m.group(1).lower().replace("question", "q")
        tok = m.group(2)
        ordv = (int(tok.split(".")[0]) if tok[0].isdigit()
                else roman_to_int(tok) or (ord(tok.upper()) - ord("A") + 1))
        return Marker(f"word:{word}", ordv, 0, m.group(0).strip())
    m = _LETTER_DOTTED.match(text)
    if m:
        return Marker("letterdotted", ord(m.group(1)) - ord("A") + 1,
                      m.group(2).count(".") + 1, m.group(0).strip())
    # paren forms before dotted: "1)" must be paren_digit, not dotted
    m = _PAREN_ROMAN.match(text)
    if m and roman_to_int(m.group(1)):
        return Marker("paren_roman", roman_to_int(m.group(1)), 0, m.group(0).strip())
    m = _PAREN_ALPHA.match(text)
    if m:
        ch = m.group(1) or m.group(2)
        return Marker("paren_alpha", ord(ch) - ord("a") + 1, 0, m.group(0).strip())
    m = _PAREN_DIGIT.match(text)
    if m:
        return Marker("paren_digit", int(m.group(1)), 0, m.group(0).strip())
    m = _DOTTED.match(text)
    if m:
        return Marker("dotted", int(m.group(1).split(".")[0]),
                      m.group(1).count("."), m.group(0).strip())
    m = _ALPHA_DOT.match(text)
    if m:
        tok = m.group(1)
        rv = roman_to_int(tok)
        if rv is not None and (len(tok) > 1 or tok.upper() == "I"):
            # multi-char roman ("II", "IV") or a leading bare "I" — roman-leaning
            return Marker("roman", rv, 0, m.group(0).strip(),
                          ambiguous=(len(tok) == 1))
        if len(tok) == 1:
            li = ord(tok.upper()) - ord("A") + 1
            return Marker("alpha", li, 0, m.group(0).strip(),
                          ambiguous=(rv is not None))
        return None
    return None


# ---------------------------------------------------------------------------
# Schema induction (rung 1')
# ---------------------------------------------------------------------------

@dataclass
class Leveled:
    title: str
    level: int | None        # None = no deterministic level signal (route to rung 2)
    confidence: str          # high | med | low | none
    signal: str              # dotted | induced | sequence | typography | flat | ...


def _resolve_ambiguous(markers: list[Marker | None]) -> list[Marker | None]:
    """Roman/letter single tokens resolved by run continuity across the doc
    (generalises the old IEEE trick, gap-tolerant — F4 fix): a token claims the
    series whose next expected ordinal it matches or minimally exceeds."""
    out: list[Marker | None] = list(markers)
    next_ord: dict[str, int] = {}
    for i, mk in enumerate(out):
        if mk is None:
            continue
        if mk.ambiguous:
            cands: list[tuple[int, str, int]] = []   # (distance, kind, ordinal)
            tok = mk.raw.rstrip(".) ").strip()
            rv = roman_to_int(tok)
            li = (ord(tok.upper()) - ord("A") + 1) if len(tok) == 1 else None
            for kind, ordv in (("roman", rv), ("alpha", li)):
                if ordv is None:
                    continue
                exp = next_ord.get(kind, 1)
                d = ordv - exp
                if d >= 0:                    # continues (gap-tolerant)
                    cands.append((d, kind, ordv))
            if cands:
                d, kind, ordv = min(cands)
                out[i] = Marker(kind, ordv, 0, mk.raw)
            # else: keep the classifier's lean
        mk2 = out[i]
        if mk2 is not None:
            next_ord[mk2.kind] = mk2.ordinal + 1
            if mk2.kind == "roman":            # a new section resets letter runs
                next_ord["alpha"] = 1
    return out


def _restart_edges(seq: list[Marker]) -> dict[tuple[str, str], int]:
    """Evidence that kind B nests under kind A: B's ordinal returns to 1
    after an occurrence of A."""
    edges: dict[tuple[str, str], int] = {}
    last_kind: str | None = None
    seen_since: dict[str, str | None] = {}
    prev_ord: dict[str, int] = {}
    for mk in seq:
        if mk.kind in prev_ord and mk.ordinal <= prev_ord[mk.kind]:
            parent = seen_since.get(mk.kind)
            if parent and parent != mk.kind and mk.ordinal == 1:
                edges[(parent, mk.kind)] = edges.get((parent, mk.kind), 0) + 1
        prev_ord[mk.kind] = mk.ordinal
        for k in seen_since:
            seen_since[k] = mk.kind if mk.kind != k else seen_since[k]
        seen_since.setdefault(mk.kind, last_kind)
        last_kind = mk.kind
    return edges


def induce_levels(headings: list[str]) -> list[Leveled]:
    """Marker-schema induction: per-heading levels from the document's own
    numbering grammar.  Headings with no marker get level=None (rung-2 input),
    never a forced level 1 — the F3 fix."""
    markers = _resolve_ambiguous([classify_marker(h) for h in headings])
    seq = [m for m in markers if m is not None]

    kinds: list[str] = []
    for m in seq:
        if m.kind not in kinds:
            kinds.append(m.kind)

    # Nesting from restart evidence, then rank kinds: a kind with no parent is
    # top-level; children rank below their parent.  Kinds with no evidence
    # keep first-appearance order *after* evidenced placement.
    edges = _restart_edges(seq)
    parent_of: dict[str, str] = {}
    for (a, b), n in sorted(edges.items(), key=lambda kv: -kv[1]):
        if b not in parent_of and a != b:
            parent_of[b] = a

    def kind_level(k: str) -> int:
        lvl, cur, hops = 1, k, 0
        while cur in parent_of and hops < 6:
            lvl += 1; cur = parent_of[cur]; hops += 1
        return lvl

    base: dict[str, int] = {k: kind_level(k) for k in kinds}
    # No restart evidence at all and >1 kinds: fall back to first-appearance
    # order (first kind seen = top), a weak but honest prior.
    if not edges and len(kinds) > 1:
        base = {k: i + 1 for i, k in enumerate(kinds)}
    # Priors that override the weak fallback (never evidenced placements):
    # word-prefixed markers self-announce section-hood → top level; a
    # letter-dotted compound ("A.1") is inherently one below its letter.
    for k in kinds:
        if k.startswith("word:") and not any(p[1] == k for p in edges):
            base[k] = 1
    if "letterdotted" in base and not any(p[1] == "letterdotted" for p in edges):
        parent_lv = base.get("alpha") or max(
            (base[k] for k in base if k.startswith("word:")), default=1)
        base["letterdotted"] = parent_lv + 1
    # Adjacency prior: a kind with no restart evidence that is not the
    # document's opening kind sits one below the marker heading immediately
    # preceding its first occurrence ("1) Detail" right after a level-2
    # "A. Problem" → level 3).  Word-prefixed kinds keep their top-level prior.
    evidenced = {b for (_, b) in edges}
    for k in kinds[1:]:
        if k in evidenced or k.startswith("word:") or k == "letterdotted":
            continue
        first_i = next(i for i, m in enumerate(seq) if m.kind == k)
        if first_i > 0:
            prev = seq[first_i - 1]
            if prev.kind != k:
                base[k] = base.get(prev.kind, 1) + prev.depth_hint + 1

    out: list[Leveled] = []
    for h, mk in zip(headings, markers):
        if mk is None:
            out.append(Leveled(h, None, "none", "unnumbered"))
            continue
        if mk.kind == "dotted":
            depth = base.get(mk.kind, 1) + mk.depth_hint          # "1.2" → base+1
        elif mk.kind == "letterdotted":
            depth = base.get(mk.kind, 2) + mk.depth_hint - 1      # "A.1" → base
        else:
            depth = base.get(mk.kind, 1)
        conf = ("high" if mk.kind in ("dotted", "letterdotted")
                or mk.kind.startswith("word:")
                else ("med" if edges else "low"))
        sig = "dotted" if mk.kind in ("dotted", "letterdotted") else "induced"
        out.append(Leveled(h, depth, conf, sig))
    return out


# ---------------------------------------------------------------------------
# Typography calibration (rung 2)
# ---------------------------------------------------------------------------

def typography_signatures(head_blocks, pdf_path, ocr_scale: float = 2.0):
    """(size, weight) signature per heading block from the text layer.
    Returns list of tuples or None per block."""
    from collections import Counter

    import pypdfium2 as pdfium  # type: ignore[import-untyped]
    import pypdfium2.raw as R   # type: ignore[import-untyped]

    doc = pdfium.PdfDocument(str(pdf_path))
    by_page: dict[int, list] = {}
    for i, b in enumerate(head_blocks):
        by_page.setdefault(b.page, []).append((i, b))
    sigs: list[tuple[float, int] | None] = [None] * len(head_blocks)
    s = 1.0 / ocr_scale
    for page_idx, entries in by_page.items():
        page = doc[page_idx]
        ph = page.get_height()
        tp = page.get_textpage()
        chars = []
        for ci in range(tp.count_chars()):
            box = tp.get_charbox(ci)
            fs = R.FPDFText_GetFontSize(tp, ci)
            try:
                fw = R.FPDFText_GetFontWeight(tp, ci)
            except Exception:
                fw = -1
            chars.append((box, fs, fw))
        for idx, b in entries:
            x0, y0, x1, y1 = b.bbox
            bl, bb, br, bt = x0 * s, ph - y1 * s, x1 * s, ph - y0 * s
            hits = [(round(fs * 2) / 2, (fw if fw and fw > 0 else 400))
                    for (cl, cb, cr, ct), fs, fw in chars
                    if fs >= 3 and cl < br + 4 and cr > bl - 4 and cb < bt + 4 and ct > bb - 4]
            if hits:
                sz = Counter(h[0] for h in hits).most_common(1)[0][0]
                wt = Counter(h[1] for h in hits).most_common(1)[0][0]
                sigs[idx] = (sz, int(wt))
    return sigs


def calibrate_typography(leveled: list[Leveled],
                         sigs: list[tuple[float, int] | None]) -> list[Leveled]:
    """Fill level=None headings from typography clusters.

    Calibrated: a cluster containing headings with induced levels adopts their
    majority level (an unnumbered heading typeset like the level-2s IS a
    level 2 — the F3/mixed-mode fix).  Uncalibrated clusters rank by
    (size desc, weight desc) below/between the known ones; with NO induced
    levels anywhere (fully unnumbered doc), cluster rank = level, and a
    single cluster (or no signatures) degrades to flat — never fabricated
    depth (F10/F11 guarded by clustering on the exact signature, not raw
    size ranking).
    """
    from collections import Counter, defaultdict

    out = list(leveled)
    clusters: dict[tuple[float, int], list[int]] = defaultdict(list)
    for i, sg in enumerate(sigs):
        if sg is not None:
            clusters[sg].append(i)

    known: dict[tuple[float, int], int] = {}
    for sg, idxs in clusters.items():
        lv = [out[i].level for i in idxs if out[i].level is not None]
        if lv:
            known[sg] = Counter(lv).most_common(1)[0][0]

    ranked = sorted(clusters, key=lambda sg: (-sg[0], -sg[1]))
    if not known:
        rank_level = {sg: r + 1 for r, sg in enumerate(ranked)}
    else:
        rank_level = dict(known)
        deepest = max(known.values())
        for sg in ranked:
            if sg not in rank_level:
                smaller = [known[k] for k in known if k[0] > sg[0]]
                rank_level[sg] = (max(smaller) + 1) if smaller else deepest + 1

    for sg, idxs in clusters.items():
        for i in idxs:
            if out[i].level is None:
                out[i] = Leveled(out[i].title, rank_level[sg],
                                 "med" if known else "low", "typography")
    for i, l in enumerate(out):
        if l.level is None:
            out[i] = Leveled(l.title, 1, "none", "flat")
    return out


# ---------------------------------------------------------------------------
# Document-level confidence gate (rung 3)
# ---------------------------------------------------------------------------

def gate(leveled: list[Leveled], threshold: float = 0.8) -> tuple[bool, float]:
    """(resolved, share of high/med confidence).  Resolved documents ship
    deterministically; the residue is rung-4 (small-LM) input."""
    if not leveled:
        return True, 1.0
    ok = sum(1 for l in leveled if l.confidence in ("high", "med"))
    share = ok / len(leveled)
    return share >= threshold, share
