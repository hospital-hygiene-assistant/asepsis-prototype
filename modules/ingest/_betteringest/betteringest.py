"""
Vendored BetterIngester ingest/betteringest.py — the deterministic
PDF → structure-faithful-markdown + managed-assets pipeline, adapted for
astepsis.  Deviations from upstream (all documented inline):

  · package-relative imports (astepsis has its own top-level `ingest.py`,
    so upstream's `ingest.*` package name cannot be used here);
  · `Asset.bbox` — the crop's bounding box (render px at ocr_scale) is kept
    from the save_asset_crops manifest so pins can point into the source PDF;
  · `BetterIngest(cache_dir=…)` — the OCR cache location is explicit instead
    of cwd-relative (same cache format; BetterIngester's own caching is
    reused, keyed by (pdf sha, config));
  · `describe_assets` takes a REQUIRED `chat` callable — upstream defaulted to
    run_pipeline._api_chat, which is part of BetterIngester's benchmark
    harness and out of scope here.  astepsis supplies the callable from
    modules/ingest/_captioning.py (local Ollama by default);
  · `to_pageindex()` is not vendored: it drives the vendored PageIndex
    checkout (vendor/pageindex) which astepsis does not carry — astepsis's own
    pageindex.py builds the tree from the massaged markdown instead.  The
    page-anchor metadata it would attach comes from the same helpers, vendored
    in anchors.py.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path

from .tree import normalise_title
from .ladder import build_ladder_tree
from .ocr import Block, OcrConfig, run_ocr
from .reconstruct import (
    assign_levels, build_content_tree, font_sizes_for_headings, ocr_doc_title,
    reading_order, recovered_reachability, save_asset_crops, to_markdown,
)
from .tree import Node


def find_breadcrumb(node: Node, target_norm: str, path: list[str] = None) -> list[str] | None:
    if path is None:
        path = []
    current_norm = normalise_title(node.title)
    current_path = path + [node.title] if node.title else path
    if current_norm == target_norm:
        return current_path
    for child in node.children:
        res = find_breadcrumb(child, target_norm, current_path)
        if res is not None:
            return res
    return None


@dataclass
class Asset:
    asset_id: str          # "figure_1", "table_2" — stable direct reference
    type: str              # "figure" | "table"
    number: int            # rendered number ("Figure 3" → 3)
    caption: str
    page: int              # 1-based physical page (PageIndex convention)
    image: str             # saved crop PNG path
    sections: list[str] = field(default_factory=list)   # normalised citing headings
    description: str = ""  # multimodal description (describe_assets)
    bbox: list[float] = field(default_factory=list)     # crop bbox, render px @ ocr_scale
    table_markdown: str = ""
    custom_name: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class IngestedDoc:
    doc_name: str
    title: str
    pdf_path: str
    md_path: str
    markdown: str
    assets: list[Asset]
    tree: object                  # content-at-leaves tree (ladder hierarchy)
    blocks: list[Block]
    ladder_diag: dict
    ocr_scale: float


def _ladder_levels(diag: dict,
                   head_titles: list[str]) -> list[int | None] | None:
    """Map the ladder's final leveling onto the detected headings, one entry
    per heading in reading order.  Unmapped headings get None (caller fills
    them from the legacy assigner); a fully unusable diag returns None.

    Rungs 1-5: diag['headings'] IS the detected-heading list (positional).
    Rung 0: diag['headings'] holds validated OUTLINE entries, which may omit
    some detected headings (references/acknowledgements are common) — match
    by normalised title in order (queue-consuming, so duplicate titles
    resolve by position), None where the outline is silent."""
    heads = diag.get("headings") or []
    if not heads or not head_titles:
        return None
    if diag.get("rung") not in (0,) and len(heads) == len(head_titles):
        return [h[1] for h in heads]
    queue = [(normalise_title(t), lv) for t, lv, *_ in heads]
    levels: list[int | None] = []
    for title in head_titles:
        key = normalise_title(title)
        for i, (qt, lv) in enumerate(queue):
            if qt == key:
                levels.append(lv)
                del queue[:i + 1]
                break
        else:
            levels.append(None)
    return levels if any(lv is not None for lv in levels) else None


class BetterIngest:
    """Deterministic PDF → structure-faithful markdown + managed assets.
    One instance is reusable across documents."""

    def __init__(self, out_dir: str | Path = "out", ocr_scale: float = 2.0,
                 ladder_llm=None, outline_demotes: bool = False,
                 cache_dir: str | Path | None = None):
        self.out_dir = Path(out_dir)
        self.ocr_scale = ocr_scale
        self.ladder_llm = ladder_llm    # optional rung-4 tiebreak callback
        # astepsis: explicit OCR cache location (None → upstream default
        # `.ocr_cache` relative to cwd).  Same cache format either way.
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        # rung-0 docs: what happens to headings the validated outline disowns.
        # False (default): levelled by the legacy assigner (upstream-measured
        # to keep content_f1 bit-identical to the benchmarked pipeline).
        self.outline_demotes = outline_demotes

    def ingest(self, pdf_path: str | Path) -> IngestedDoc:
        pdf = Path(pdf_path)
        name = pdf.stem
        out = self.out_dir / name
        out.mkdir(parents=True, exist_ok=True)

        ocr_kwargs = {} if self.cache_dir is None else {"cache_dir": self.cache_dir}
        blocks = run_ocr(pdf, OcrConfig(ocr_scale=self.ocr_scale), **ocr_kwargs)
        title = ocr_doc_title(blocks, name)

        # hierarchy from the ladder; the content partition carries it whenever
        # the ladder's leveling can be mapped onto the detected headings
        # (positional for rungs 1-5, title-matched for the rung-0 outline).
        _, diag = build_ladder_tree(blocks, title, pdf_path=pdf,
                                    ocr_scale=self.ocr_scale,
                                    llm=self.ladder_llm)
        head_blocks = [b for b in reading_order(blocks)
                       if b.label == "paragraph_title"]
        mapped = _ladder_levels(diag, [b.text for b in head_blocks])
        levels = None
        if mapped is not None and any(lv is None for lv in mapped):
            if self.outline_demotes and diag.get("rung") == 0:
                # validated outline disowns these headings → demote to body
                pass
            else:                                   # gaps → legacy fill
                sizes = font_sizes_for_headings(head_blocks, pdf,
                                                self.ocr_scale)
                legacy = assign_levels([b.text for b in head_blocks], sizes)
                mapped = [m if m is not None else lg
                          for m, lg in zip(mapped, legacy)]
        levels = mapped
        tree = build_content_tree(blocks, title, pdf, self.ocr_scale,
                                  levels=levels)

        # assets: crops + reachability (which sections cite which asset)
        manifest = save_asset_crops(blocks, pdf, out / "assets", self.ocr_scale)
        cites: dict[tuple[str, int], list[str]] = {}
        for h, t, num in recovered_reachability(blocks, pdf, self.ocr_scale):
            cites.setdefault((t, num), []).append(h)
        assets = []
        seen: dict[str, int] = {}
        for m in manifest:
            aid = f"{m['type']}_{m['number']}"
            if aid in seen:                     # duplicate rendered numbers
                seen[aid] += 1
                aid = f"{aid}_{seen[aid]}"
            else:
                seen[aid] = 1
            
            sections = sorted(cites.get((m["type"], m["number"]), []))
            if not sections and m.get("physical_section"):
                norm_phys = normalise_title(m["physical_section"])
                if norm_phys:
                    sections = [norm_phys]

            assets.append(Asset(
                asset_id=aid, type=m["type"], number=m["number"],
                caption=m["caption"], page=int(m["page"]) + 1,
                image=m["image"],
                sections=sections,
                bbox=list(m.get("bbox") or [])))

        # markdown with asset links under their citing sections
        at: dict[str, list[str]] = {}
        linked: set[str] = set()
        for a in assets:
            link = f"![{a.type} {a.number}]({Path(a.image).relative_to(out)})"
            for h in a.sections:
                at.setdefault(h, []).append(link)
                linked.add(a.asset_id)
            if a.asset_id not in linked:
                at.setdefault("\x00unplaced", []).append(link)
        markdown = to_markdown(tree, at)
        md_path = out / f"{name}.md"
        md_path.write_text(markdown, encoding="utf-8")

        return IngestedDoc(doc_name=name, title=title, pdf_path=str(pdf),
                           md_path=str(md_path), markdown=markdown,
                           assets=assets, tree=tree, blocks=blocks,
                           ladder_diag=diag, ocr_scale=self.ocr_scale)

    def finalize_review_doc(self, pdf_path: str | Path, confirmed_blocks: list[Block], non_asset_blocks: list[Block]) -> IngestedDoc:
        pdf = Path(pdf_path)
        name = pdf.stem
        out = self.out_dir / name
        out.mkdir(parents=True, exist_ok=True)

        blocks = non_asset_blocks + confirmed_blocks
        title = ocr_doc_title(blocks, name)

        _, diag = build_ladder_tree(blocks, title, pdf_path=pdf,
                                    ocr_scale=self.ocr_scale,
                                    llm=self.ladder_llm)
        head_blocks = [b for b in reading_order(blocks)
                       if b.label == "paragraph_title"]
        mapped = _ladder_levels(diag, [b.text for b in head_blocks])
        levels = None
        if mapped is not None and any(lv is None for lv in mapped):
            if self.outline_demotes and diag.get("rung") == 0:
                pass
            else:
                sizes = font_sizes_for_headings(head_blocks, pdf,
                                                self.ocr_scale)
                legacy = assign_levels([b.text for b in head_blocks], sizes)
                mapped = [m if m is not None else lg
                          for m, lg in zip(mapped, legacy)]
        levels = mapped
        tree = build_content_tree(blocks, title, pdf, self.ocr_scale,
                                  levels=levels)

        # assets: crops + reachability
        manifest = save_asset_crops(blocks, pdf, out / "assets", self.ocr_scale)
        cites: dict[tuple[str, int], list[str]] = {}
        for h, t, num in recovered_reachability(blocks, pdf, self.ocr_scale):
            cites.setdefault((t, num), []).append(h)
        assets = []
        seen: dict[str, int] = {}
        for m in manifest:
            aid = f"{m['type']}_{m['number']}"
            if aid in seen:
                seen[aid] += 1
                aid = f"{aid}_{seen[aid]}"
            else:
                seen[aid] = 1
            
            sections = sorted(cites.get((m["type"], m["number"]), []))
            if not sections and m.get("physical_section"):
                norm_phys = normalise_title(m["physical_section"])
                if norm_phys:
                    sections = [norm_phys]

            # Breadcrumb auto-naming or custom override
            caption = m.get("custom_name")
            if not caption:
                if sections:
                    breadcrumb = find_breadcrumb(tree, sections[0])
                    if breadcrumb:
                        parts = breadcrumb[1:] if len(breadcrumb) > 1 else breadcrumb
                        breadcrumb_str = " / ".join(parts)
                        caption = f"{breadcrumb_str} / {m['type'].capitalize()} {m['number']}"
            if not caption:
                caption = m["caption"]

            assets.append(Asset(
                asset_id=aid, type=m["type"], number=m["number"],
                caption=caption, page=int(m["page"]) + 1,
                image=m["image"],
                sections=sections,
                bbox=list(m.get("bbox") or []),
                table_markdown=m.get("table_markdown", ""),
                custom_name=m.get("custom_name")))

        # markdown with asset links
        at: dict[str, list[str]] = {}
        linked: set[str] = set()
        for a in assets:
            link = f"![{a.type} {a.number}]({Path(a.image).relative_to(out)})"
            for h in a.sections:
                at.setdefault(h, []).append(link)
                linked.add(a.asset_id)
            if a.asset_id not in linked:
                at.setdefault("\x00unplaced", []).append(link)
        markdown = to_markdown(tree, at)
        md_path = out / f"{name}.md"
        md_path.write_text(markdown, encoding="utf-8")

        return IngestedDoc(doc_name=name, title=title, pdf_path=str(pdf),
                           md_path=str(md_path), markdown=markdown,
                           assets=assets, tree=tree, blocks=blocks,
                           ladder_diag=diag, ocr_scale=self.ocr_scale)

    # ── optional multimodal step ───────────────────────────────────────────────

    DESCRIBE_PROMPT = (
        "This is a {type} from a document (caption: {caption!r}). Describe "
        "factually what it shows in 2-3 sentences for a retrieval index: "
        "variables/axes or columns, trends or key values, and what a reader "
        "could look up in it. No preamble.")

    def describe_assets(self, doc: IngestedDoc, chat,
                        max_chars: int = 600, progress_cb=None) -> IngestedDoc:
        """Give every asset a retrieval-oriented description via a multimodal
        model (crop PNG + caption).  `chat(messages) -> (text, …) | text | None`
        receives OpenAI-style messages with a data-URL image part.  Assets
        whose call fails keep description="" (retry = rerun).

        astepsis deviation: `chat` is required — upstream defaulted to the
        benchmark harness's cached transport (run_pipeline._api_chat), which
        is out of scope here.  See modules/ingest/_captioning.py for the
        pluggable backends astepsis provides."""
        total_missing = sum(1 for a in doc.assets if not a.description)
        idx_missing = 0
        for a in doc.assets:
            if a.description:
                continue
            if progress_cb:
                try:
                    progress_cb(idx_missing, total_missing)
                except Exception:
                    pass
            idx_missing += 1
            if a.type == "table" and getattr(a, "table_markdown", ""):
                prompt = (
                    f"Analyze this table (formatted in markdown) and write a factual description of what it shows in 2-3 sentences.\n"
                    f"Include the variables/axes or columns, key trends or values, and what a reader could look up in it. No preamble.\n\n"
                    f"Table:\n{a.table_markdown}"
                )
                messages = [{"role": "user", "content": prompt}]
            else:
                b64 = base64.b64encode(Path(a.image).read_bytes()).decode()
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": self.DESCRIBE_PROMPT.format(
                        type=a.type, caption=a.caption)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]}]
            r = chat(messages)
            if r is not None:
                text = r[0] if isinstance(r, (tuple, list)) else r
                text = re.sub(r"<thought>.*?(</thought>|$)", "", text,
                              flags=re.S)
                a.description = " ".join(text.split())[:max_chars]
        return doc
