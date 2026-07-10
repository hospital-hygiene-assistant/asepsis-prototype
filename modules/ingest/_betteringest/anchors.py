"""
Vendored subset of BetterIngester's ingest/pageindex_vendor.py — only the
deterministic page-anchor helpers that need no LLM transport and no vendored
PageIndex checkout.  These are the exact functions IngestedDoc.to_pageindex()
uses to enrich the tree with physical_index page anchors; astepsis's massager
reuses them so provenance comes from the same source of truth instead of being
re-extracted.

Source: BetterIngester ingest/pageindex_vendor.py (verbatim excerpts).
"""
from __future__ import annotations

from .tree import normalise_title


def heading_pages_from_blocks(blocks) -> list[tuple[str, int]]:
    """(normalised heading, 1-based page) in reading order, from our layout
    blocks — the page source for attach_page_anchors."""
    out = []
    for b in blocks:
        if b.label in ("doc_title", "paragraph_title") and b.text.strip():
            out.append((normalise_title(b.text), int(b.page) + 1))
    return out


def attach_page_anchors(structure: list[dict],
                        heading_pages: list[tuple[str, int]]) -> list[dict]:
    """Add PDF-style `physical_index` to an md-path tree in place.

    Nodes are visited in document (pre-)order and matched against the ordered
    heading/page list; each match consumes the queue up to that heading, so
    repeated titles resolve by position, not just name."""
    queue = list(heading_pages)

    def walk(nodes: list[dict]) -> None:
        for d in nodes:
            t = normalise_title(str(d.get("title", "")))
            for i, (ht, pg) in enumerate(queue):
                if ht == t:
                    d["physical_index"] = pg
                    del queue[:i + 1]
                    break
            walk(d.get("nodes") or [])

    walk(structure)
    return structure
