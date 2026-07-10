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
