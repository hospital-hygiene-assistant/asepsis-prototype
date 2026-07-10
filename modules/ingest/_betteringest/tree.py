"""
Vendored subset of BetterIngester's bench/tree.py — only the shared tree data
structure and title normalisation that the ingestion pipeline itself depends
on.  The scoring half of bench/tree.py (benchmark-only) is deliberately left
behind.

Source: BetterIngester bench/tree.py (verbatim excerpts).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Node:
    title: str
    children: list["Node"] = field(default_factory=list)
    # Phase-3 additions (default to the old heading-only behaviour):
    kind: str = "heading"          # "heading" (internal) | "content" (leaf) | "asset"
    content: str = ""              # body text — content leaves only
    asset_type: str = ""           # "figure" | "table" — asset nodes only
    label: str = ""                # \label key — asset nodes (target of \ref)
    refs: list[str] = field(default_factory=list)  # content leaves: \label keys referenced

    def all_nodes(self) -> list["Node"]:
        out = [self]
        for c in self.children:
            out.extend(c.all_nodes())
        return out

    def content_leaves(self) -> list["Node"]:
        return [n for n in self.all_nodes() if n.kind == "content"]


def normalize_to_leaves(node: "Node") -> "Node":
    """Enforce: content lives only in leaves.  Any node that has children *and*
    its own content gets that content pushed into a synthetic leaf child (titled
    after the node), so internal nodes are pure structure and every piece of text
    is addressable as a leaf.  Mutates and returns the tree."""
    for c in node.children:
        normalize_to_leaves(c)
    if node.children:                              # internal node → structure only
        if node.content.strip() or node.refs:
            node.children.insert(0, Node(
                title=node.title, kind="content",
                content=node.content, refs=list(node.refs)))
        node.content, node.refs, node.kind = "", [], "heading"
    elif node.content.strip() or node.refs:        # leaf with text → content leaf
        node.kind = "content"
    return node


# ---------------------------------------------------------------------------
# Title normalisation (evaluation-schema fixes)
# ---------------------------------------------------------------------------

_MD_EMPHASIS = re.compile(r"[*`]+")
# Fold Unicode punctuation variants to ASCII so a heading isn't counted as both a
# miss and a false positive purely because an OCR/LLM emitted a smart quote —
# an evaluation-schema artifact, not a content difference.
_PUNCT_FOLD = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "‐": "-", " ": " ",
})
# Appendix sections render with an auto letter ("A PEEU Prompt", "B AHC057 …")
# while the LaTeX truth is just "PEEU Prompt".  Strip a leading single capital
# letter followed by a CAPITAL or digit (so the appendix marker goes) — but never
# when the next char is lowercase, so a real "A problem"/"An overview" survives.
# Runs on the original-case string, before lower-casing.
_APPENDIX_PREFIX = re.compile(r"^[A-Z]\s+(?=[A-Z0-9])")
# IEEE-style compound bookmark prefixes from embedded PDF outlines: "II Datacenter
# architecture", "I-A Related research", "III-B2 Details".  Multi-char roman
# (optionally dash-joined sub-tokens), followed by a capitalised/digit word —
# the same next-word guard as the appendix rule, so "I am legend" survives.
# Runs on the original-case string, before lower-casing.
_OUTLINE_PREFIX = re.compile(r"^[IVXLCDM]{2,7}(?:-[A-Z][0-9]*)*\s+(?=[A-Z0-9])"
                             r"|^[IVXLCDM]{1,7}(?:-[A-Z][0-9]*)+\s+(?=[A-Z0-9])")
# leading rendered numbering. Digit numbering needs no trailing separator
# ("1 Introduction", "1.1 Background"); letter/roman numbering requires a "." or
# ")" so a real article like "A problem" is never stripped.  Letter-dotted-digit
# compounds ("A.1 Data") are appendix subsections and are stripped.
_ENUM_PREFIX = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?|[ivxlcdm]+[.)]|[a-z][.)]|[a-z]\.\d+(?:\.\d+)*[.)]?)\s+")


def normalise_title(text: str) -> str:
    """Canonical form for comparing headings across LaTeX vs recovered markdown."""
    text = text.translate(_PUNCT_FOLD).strip()
    text = _OUTLINE_PREFIX.sub("", text)
    text = _APPENDIX_PREFIX.sub("", text)
    text = _MD_EMPHASIS.sub("", text.lower())
    text = " ".join(text.split())
    text = _ENUM_PREFIX.sub("", text)
    return text.strip(" .,:;-–—#")
