"""
ingest/ladder.py — the confidence-routed escalation ladder (assembly).

Routes each document through deterministic rungs and escalates only measured
residual uncertainty (docs/heuristic-robustness.md §5):

  rung -1  no detected headings → root-leaf document (F8/F9 path)
  rung 0   embedded outline, per-entry validation gate  (ingest/outline.py)
  rung 1'  marker-schema induction                      (ingest/levels.py)
  rung 2   typography clusters, calibrated              (ingest/levels.py)
  rung 3   confidence gate: resolved? → ship deterministically
  rung 4   small-LM tiebreak on the compact heading list (optional callback)
  rung 5   flat + confidence none — the honest floor (implicit: rung 1-2
           output ships even under the gate when no LLM is provided)

`build_ladder_tree` returns the tree plus a diagnostics dict: which rung
resolved the document, per-heading (title, level, confidence, signal), the
gate share, and the outline verdict — the rung firing distribution is itself
a reported result.
"""

from __future__ import annotations

from typing import Callable

from .tree import Node
from .levels import Leveled, calibrate_typography, gate, induce_levels, \
    typography_signatures
from .outline import outline_tree, read_outline, validate_outline
from .reconstruct import reading_order

_SECTION_TITLE = "paragraph_title"

# Optional LLM tiebreak: callback(prompt) -> raw text (JSON expected), or None.
LlmFn = Callable[[str], str | None]


def _tree_from_leveled(leveled: list[Leveled], doc_title: str) -> Node:
    root = Node(title=doc_title)
    stack: list[tuple[int, Node]] = [(0, root)]
    for l in leveled:
        lv = l.level if l.level is not None else 1
        node = Node(title=l.title)
        while stack and stack[-1][0] >= lv:
            stack.pop()
        (stack[-1] if stack else (0, root))[1].children.append(node)
        stack.append((lv, node))
    return root


def _tiebreak_prompt(leveled: list[Leveled],
                     sigs: list[tuple[float, int] | None]) -> str:
    lines = []
    for l, sg in zip(leveled, sigs or [None] * len(leveled)):
        ty = f" font={sg[0]}pt,w{sg[1]}" if sg else ""
        lines.append(f"- {l.title!r} (tool guess: level {l.level}, "
                     f"confidence {l.confidence}{ty})")
    listing = "\n".join(lines)
    return (
        "These are the detected section headings of one document, in reading "
        "order, with an automatic tool's UNCERTAIN level guesses (level 1 = "
        "top section). The guesses may be wrong — use the heading texts and "
        "your judgement.\n" + listing +
        '\n\nReturn ONLY JSON: {"levels": [1, 2, ...]} with one integer per '
        "heading, same order.\nJSON:\n")


def clean_llm_json(raw: str) -> str:
    """Best-effort JSON text from an LLM reply: drops <thought>…</thought>
    reasoning blocks (Gemma) and markdown code fences (Gemini), then returns
    the last OUTERMOST balanced {...} or [...] region — models put the answer
    last.  "Outermost" matters: for '{"levels": [1, 2]}' the whole dict is the
    answer, not the [1, 2] nested inside it (the pre-2026-07-09 version
    returned the nested array, silently breaking every dict-with-array-value
    reply — see PINS repair-4)."""
    import re
    raw = re.sub(r"<thought>.*?(</thought>|$)", "", raw, flags=re.S)
    raw = re.sub(r"```(?:json)?", "", raw)
    spans = []
    for open_c, close_c in (("{", "}"), ("[", "]")):
        depth = 0
        start = None
        for i, ch in enumerate(raw):
            if ch == open_c:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == close_c and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append((start, i + 1))
    if not spans:
        return raw.strip()
    # outermost regions only, then the last one in reading order
    outer = [s for s in spans
             if not any(o[0] < s[0] and s[1] <= o[1] for o in spans if o != s)]
    start, end = max(outer, key=lambda s: (s[1], -s[0]))
    return raw[start:end]


def _parse_tiebreak(raw: str, n: int) -> list[int] | None:
    import json
    try:
        data = json.loads(clean_llm_json(raw))
        # tolerate a bare [1, 2, ...] reply alongside {"levels": [...]}
        levels = data if isinstance(data, list) else data.get("levels")
    except Exception:
        return None
    if (isinstance(levels, list) and len(levels) == n
            and all(isinstance(x, int) and 1 <= x <= 6 for x in levels)):
        return levels
    return None


def build_ladder_tree(blocks, doc_title: str = "", pdf_path=None,
                      ocr_scale: float = 2.0,
                      llm: LlmFn | None = None) -> tuple[Node, dict]:
    """The ladder: (tree, diagnostics).  Deterministic unless `llm` is given
    AND the confidence gate fails."""
    diag: dict = {"rung": None, "outline": None, "gate_share": None,
                  "headings": [], "llm_used": False}

    head_blocks = [b for b in reading_order(blocks) if b.label == _SECTION_TITLE]

    # rung -1 — nothing to level
    if not head_blocks:
        diag["rung"] = -1
        return Node(title=doc_title), diag

    # rung 0 — embedded outline
    if pdf_path is not None:
        entries = read_outline(pdf_path)
        v = validate_outline(entries, blocks)
        diag["outline"] = {"verdict": v.verdict, "match_rate": round(v.match_rate, 3),
                           "notes": v.notes}
        if v.verdict == "full":
            diag["rung"] = 0
            diag["headings"] = [(e.title, e.level + 1, "high", "outline")
                                for e in v.entries]
            return outline_tree(v.entries, doc_title), diag
        # titles_only: keep detection (which has bboxes); levels from below.

    # rung 1' — marker-schema induction
    headings = [b.text for b in head_blocks]
    leveled = induce_levels(headings)

    # rung 2 — typography calibration for the level=None residue
    sigs = None
    if pdf_path is not None and any(l.level is None for l in leveled):
        sigs = typography_signatures(head_blocks, pdf_path, ocr_scale)
        leveled = calibrate_typography(leveled, sigs)
    else:
        leveled = calibrate_typography(leveled, [None] * len(leveled))

    # rung 3 — confidence gate
    resolved, share = gate(leveled)
    diag["gate_share"] = round(share, 3)
    diag["headings"] = [(l.title, l.level, l.confidence, l.signal) for l in leveled]
    if resolved or llm is None:
        diag["rung"] = 2 if any(l.signal == "typography" for l in leveled) else 1
        return _tree_from_leveled(leveled, doc_title), diag

    # rung 4 — small-LM tiebreak on the compact list
    raw = llm(_tiebreak_prompt(leveled, sigs or [None] * len(leveled)))
    if raw:
        fixed = _parse_tiebreak(raw, len(leveled))
        if fixed:
            diag["rung"] = 4
            diag["llm_used"] = True
            leveled = [Leveled(l.title, lv, "med", "llm-tiebreak")
                       for l, lv in zip(leveled, fixed)]
            diag["headings"] = [(l.title, l.level, l.confidence, l.signal)
                                for l in leveled]
            return _tree_from_leveled(leveled, doc_title), diag
    diag["rung"] = 5                                   # LLM failed → honest floor
    return _tree_from_leveled(leveled, doc_title), diag
