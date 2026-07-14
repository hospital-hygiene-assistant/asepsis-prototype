"""
Module 1 — Ingest: BetterIngest PDF
Runs a flat folder of real PDFs through the vendored BetterIngester pipeline
(deterministic layout OCR → escalation-ladder hierarchy → structure-faithful
markdown + asset crops), then massages the markdown with provenance pins and
asset leaves before writing it to knowledge_base/.

Where basic_markdown copies pre-cleaned .md files, this module reconstructs
markdown FROM the PDFs, and every heading/asset carries a ```pin block with
its page, bbox, and (for assets) crop image — the provenance chain that lets
the UI jump from a retrieved chunk to its exact spot in the source PDF.

The source folder is chosen in the app (native folder dialog) and persisted in
.betteringest.json; `run()` can also be pointed at a folder directly.  OCR
results are cached by BetterIngester's own (pdf sha, config) cache in
.ocr_cache/, so re-runs are fast and byte-deterministic.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

MODULE_INFO = {
    "stage": "ingest",
    "name": "betteringest_pdf",
    "label": "BetterIngest PDF",
    "description": (
        "Ingests a folder of PDFs via the BetterIngester pipeline: layout OCR "
        "(PP-DocLayoutV3), deterministic heading hierarchy, figure/table crops "
        "with captions, and provenance pins (page + bbox) on every section and "
        "asset. Local and deterministic; asset captioning uses a local Ollama "
        "vision model by default."),
    # Tells the frontend this module needs a source folder of PDFs picked
    # before it can run (triggers the folder-picker popup).
    "source": "pdf_folder",
}

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "knowledge_base"
OUT_DIR = ROOT / ".betteringest_out"        # BetterIngest working area (crops, raw md)
OCR_CACHE_DIR = ROOT / ".ocr_cache"         # BetterIngester's own OCR cache format
CONFIG_PATH = ROOT / ".betteringest.json"
SOURCES_MANIFEST = KB_DIR / ".sources.json"


# ── source-folder config ─────────────────────────────────────────────────────

def get_source_dir() -> str | None:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("source_dir")
    except Exception:
        return None


def set_source_dir(path: str) -> None:
    CONFIG_PATH.write_text(json.dumps({"source_dir": str(path)}, indent=2),
                           encoding="utf-8")


def _load_sources() -> dict:
    try:
        return json.loads(SOURCES_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── the ingest stage ─────────────────────────────────────────────────────────

def run(source_dir: str | None = None, progress=None,
        caption_backend: str | None = None,
        caption_model: str | None = None) -> dict:
    """Ingest every *.pdf in the source folder into knowledge_base/.

    `progress(info: dict)` (optional) receives {phase, doc, done, total,
    message} updates — the server threads this into /api/ingest/progress.
    Returns {"docs": [stems], "warnings": [str]}.
    """
    src = Path(source_dir or get_source_dir() or "")
    if not src or not src.is_dir():
        raise FileNotFoundError(
            f"BetterIngest source folder not set or missing: '{src}'. Pick a "
            "folder of PDFs in the app (or pass source_dir).")
    if source_dir:
        set_source_dir(str(src))

    pdfs = sorted(src.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {src}/")

    return _ingest_pdfs(pdfs, progress, caption_backend, caption_model)


def run_paths(paths: list[str], progress=None,
              caption_backend: str | None = None,
              caption_model: str | None = None) -> dict:
    """Additive ingest: process an explicit list of PDF files and/or folders
    ON TOP of whatever is already in knowledge_base/ (existing documents are
    untouched; a re-ingested stem is overwritten). Powers the Library's
    “＋ Add PDFs” affordance."""
    pdfs: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
        elif p.is_dir():
            pdfs.extend(sorted(p.glob("*.pdf")))
        else:
            raise FileNotFoundError(f"Not a PDF file or folder: {p}")
    if not pdfs:
        raise FileNotFoundError("No PDF files found in the given path(s).")
    return _ingest_pdfs(pdfs, progress, caption_backend, caption_model)


def _ingest_pdfs(pdfs: list[Path], progress=None,
                 caption_backend: str | None = None,
                 caption_model: str | None = None) -> dict:
    import os

    # Pre-flight: the layout model needs the Paddle engine.  Fail with a clear,
    # actionable message instead of a deep stack trace (flagged, not silently
    # downgraded — there is no non-OCR fallback worth having).
    try:
        import paddle  # noqa: F401
        import paddlex  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "betteringest_pdf needs paddlepaddle + paddlex (PP-DocLayoutV3 "
            f"layout model), missing from this Python ({sys.executable}): {exc}. "
            "Launch the app with the Python env that has them (pyenv 3.12.9 on "
            "this machine — see .python-version) or `pip install paddlepaddle "
            "paddlex paddleocr`.") from exc

    from modules.ingest._betteringest import BetterIngest
    from modules.ingest._captioning import CaptioningUnavailable, caption_assets
    from modules.ingest._massage import massage

    backend = caption_backend or os.environ.get("BETTERINGEST_CAPTION_BACKEND",
                                                "ollama")
    KB_DIR.mkdir(exist_ok=True)
    bi = BetterIngest(out_dir=OUT_DIR, cache_dir=OCR_CACHE_DIR)
    sources = _load_sources()
    warnings: list[str] = []
    done_stems: list[str] = []

    def report(phase: str, doc: str, i: int, message: str = ""):
        info = {"phase": phase, "doc": doc, "done": i, "total": len(pdfs),
                "message": message}
        if progress:
            progress(info)

    for i, pdf in enumerate(pdfs):
        stem = pdf.stem
        report("ocr", stem, i, f"Reconstructing {pdf.name} (layout OCR + hierarchy)…")
        doc = bi.ingest(pdf)

        report("captions", stem, i, f"Captioning {len(doc.assets)} asset(s) of {pdf.name}…")
        try:
            caption_assets(bi, doc, backend=backend, model=caption_model,
                           cache_dir=OUT_DIR)
        except CaptioningUnavailable as exc:
            # Flagged, not silently downgraded: the asset leaves still carry
            # their captions, but the user is told descriptions are missing.
            warnings.append(f"{stem}: asset descriptions skipped — {exc}")
            print(f"  [warn] {warnings[-1]}", file=sys.stderr)

        report("massage", stem, i, f"Writing {stem}.md with provenance pins…")
        asset_dir = KB_DIR / "assets" / stem
        markdown = massage(doc, stem, asset_url_base=f"/assets/{stem}")
        if doc.assets:
            asset_dir.mkdir(parents=True, exist_ok=True)
            for a in doc.assets:
                shutil.copy2(a.image, asset_dir / Path(a.image).name)
        (KB_DIR / f"{stem}.md").write_text(markdown, encoding="utf-8")

        sources[stem] = {
            "pdf": str(pdf.resolve()),
            "ocr_scale": doc.ocr_scale,
            "module": MODULE_INFO["name"],
            "assets": [a.to_dict() for a in doc.assets],
        }
        SOURCES_MANIFEST.write_text(
            json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
        done_stems.append(stem)
        print(f"  {pdf.name} → {KB_DIR / (stem + '.md')} "
              f"({len(doc.assets)} assets)")
        report("done-doc", stem, i + 1)

    print(f"\nIngested {len(done_stems)} PDF(s) into {KB_DIR}/")
    return {"docs": done_stems, "warnings": warnings}


def prepare_layout_review(pdf_paths: list[str], progress_cb=None) -> list[dict]:
    import pypdfium2 as pdfium
    from modules.ingest._betteringest.ocr import run_ocr, OcrConfig
    from modules.ingest._betteringest import BetterIngest
    from modules.ingest._betteringest.reconstruct import (
        ocr_doc_title, reading_order, assign_levels,
        font_sizes_for_headings, block_text
    )
    from modules.ingest._betteringest.ladder import build_ladder_tree
    from modules.ingest._betteringest.betteringest import _ladder_levels

    bi = BetterIngest(out_dir=OUT_DIR, cache_dir=OCR_CACHE_DIR)
    results = []
    for path_str in pdf_paths:
        pdf = Path(path_str).resolve()
        stem = pdf.stem
        
        cb = None
        if progress_cb:
            cb = lambda page, total: progress_cb(page, total, stem)
            
        blocks = run_ocr(pdf, OcrConfig(ocr_scale=bi.ocr_scale), cache_dir=bi.cache_dir, progress_cb=cb)
        
        doc = pdfium.PdfDocument(str(pdf))
        pages = []
        for i, pg in enumerate(doc):
            w, h = pg.get_size()
            pages.append({
                "page_index": i,
                "width": w * bi.ocr_scale,
                "height": h * bi.ocr_scale
            })
            
        title = ocr_doc_title(blocks, stem)
        _, diag = build_ladder_tree(blocks, title, pdf_path=pdf, ocr_scale=bi.ocr_scale, llm=bi.ladder_llm)
        head_blocks = [b for b in reading_order(blocks) if b.label == "paragraph_title"]
        mapped = _ladder_levels(diag, [b.text for b in head_blocks])
        if mapped is not None and any(lv is None for lv in mapped):
            sizes = font_sizes_for_headings(head_blocks, pdf, bi.ocr_scale)
            legacy = assign_levels([b.text for b in head_blocks], sizes)
            mapped = [m if m is not None else lg for m, lg in zip(mapped, legacy)]
            
        levels_map = {id(b): (mapped[idx] if (mapped and idx < len(mapped)) else 1) for idx, b in enumerate(head_blocks)}

        tp_pages = {}
        def is_scanned(b) -> bool:
            if b.label not in ("text", "paragraph_title"):
                return False
            if b.page not in tp_pages:
                pg = doc[b.page]
                tp_pages[b.page] = (pg.get_textpage(), pg.get_height())
            page_tp, ph = tp_pages[b.page]
            txt = block_text(b, page_tp, ph, bi.ocr_scale)
            return not bool(txt.strip())

        assets = []
        for b in blocks:
            if b.label in ("figure", "table", "chart"):
                assets.append({
                    "type": "table" if b.label == "table" else "figure",
                    "page": b.page,
                    "bbox": list(b.bbox)
                })
                
        non_assets = [b for b in blocks if b.label not in ("figure", "table", "chart")]

        results.append({
            "pdf_path": str(pdf),
            "stem": stem,
            "ocr_scale": bi.ocr_scale,
            "pages": pages,
            "detected_assets": assets,
            "non_assets": [
                {
                    "label": b.label,
                    "text": b.text,
                    "page": b.page,
                    "bbox": list(b.bbox),
                    "level": levels_map.get(id(b), 1) if b.label == "paragraph_title" else None,
                    "scanned": is_scanned(b)
                }
                for b in non_assets
            ]
        })
    return results


def extract_table_markdown(crop_img, recognizer) -> str:
    import numpy as np
    try:
        res = recognizer.ocr(np.array(crop_img), cls=False)
        if not res or not res[0]:
            return ""
        
        segments = []
        for line in res[0]:
            bbox, (text, conf) = line
            x = (bbox[0][0] + bbox[2][0]) / 2.0
            y = (bbox[0][1] + bbox[2][1]) / 2.0
            h = bbox[2][1] - bbox[0][1]
            segments.append({"text": text.strip(), "x": x, "y": y, "h": h})
        
        if not segments:
            return ""
            
        segments.sort(key=lambda s: s["y"])
        rows = []
        current_row = []
        last_y = None
        last_h = None
        for s in segments:
            if last_y is None:
                current_row.append(s)
                last_y = s["y"]
                last_h = s["h"]
            elif abs(s["y"] - last_y) < (last_h * 0.6):
                current_row.append(s)
            else:
                rows.append(current_row)
                current_row = [s]
                last_y = s["y"]
                last_h = s["h"]
        if current_row:
            rows.append(current_row)
            
        x_coords = sorted([s["x"] for s in segments])
        columns = []
        for x in x_coords:
            if not columns or x - columns[-1] > 60:
                columns.append(x)
                
        grid = []
        for row in rows:
            row_cells = [""] * len(columns)
            for s in row:
                col_idx = min(range(len(columns)), key=lambda idx: abs(columns[idx] - s["x"]))
                row_cells[col_idx] = s["text"]
            grid.append(row_cells)
            
        if not grid:
            return ""
            
        lines = []
        lines.append("| " + " | ".join(grid[0]) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for r in grid[1:]:
            lines.append("| " + " | ".join(r) + " |")
            
        return "\n".join(lines)
    except Exception as e:
        print(f"Error in extract_table_markdown: {e}")
        return ""


def finalize_ingestion(pdf_path: str, confirmed_assets: list[dict], non_assets: list[dict],
                       caption_backend: str | None = None,
                       caption_model: str | None = None,
                       progress_cb=None) -> dict:
    import os
    import shutil
    import pypdfium2 as pdfium
    from modules.ingest._betteringest import BetterIngest, Block
    from modules.ingest._captioning import CaptioningUnavailable, caption_assets
    from modules.ingest._massage import massage
    from modules.ingest._betteringest.ocr import _load_models

    pdf = Path(pdf_path).resolve()
    stem = pdf.stem

    bi = BetterIngest(out_dir=OUT_DIR, cache_dir=OCR_CACHE_DIR)
    
    # Lazily load pages for table markdown extraction
    doc_pdfium = pdfium.PdfDocument(str(pdf))
    page_images = {}
    
    def get_page_img(p_idx):
        if p_idx not in page_images:
            page_images[p_idx] = doc_pdfium[p_idx].render(scale=bi.ocr_scale).to_pil()
        return page_images[p_idx]

    confirmed_blocks = []
    for a in confirmed_assets:
        b = Block(label=a["type"], text="", page=a["page"], bbox=tuple(a["bbox"]), custom_name=a.get("name"))
        if a["type"] == "table":
            img = get_page_img(a["page"])
            x0, y0, x1, y1 = a["bbox"]
            crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
            
            _, recognizer = _load_models()
            b.table_markdown = extract_table_markdown(crop, recognizer)
            
        confirmed_blocks.append(b)

    non_asset_blocks = [
        Block(label=n["label"], text=n["text"], page=n["page"], bbox=tuple(n["bbox"]))
        for n in non_assets
    ]

    doc = bi.finalize_review_doc(pdf, confirmed_blocks, non_asset_blocks)

    backend = caption_backend or os.environ.get("BETTERINGEST_CAPTION_BACKEND", "ollama")
    try:
        def cb_caption(cur, total):
            if progress_cb:
                progress_cb(f"Captioning {stem}: asset {cur + 1}/{total}.")
        caption_assets(bi, doc, backend=backend, model=caption_model, cache_dir=OUT_DIR, progress_cb=cb_caption)
    except CaptioningUnavailable as exc:
        print(f"  [warn] asset descriptions skipped — {exc}")

    asset_dir = KB_DIR / "assets" / stem
    markdown = massage(doc, stem, asset_url_base=f"/assets/{stem}")
    if doc.assets:
        asset_dir.mkdir(parents=True, exist_ok=True)
        for a in doc.assets:
            shutil.copy2(a.image, asset_dir / Path(a.image).name)
    (KB_DIR / f"{stem}.md").write_text(markdown, encoding="utf-8")

    sources = _load_sources()
    sources[stem] = {
        "pdf": str(pdf.resolve()),
        "ocr_scale": doc.ocr_scale,
        "module": MODULE_INFO["name"],
        "assets": [a.to_dict() for a in doc.assets],
    }
    SOURCES_MANIFEST.write_text(
        json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "stem": stem,
        "pdf_path": str(pdf),
        "assets_count": len(doc.assets)
    }
