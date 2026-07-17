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
import mimetypes
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
        "and provenance geometry. Figures and tables require local operator "
        "review before one atomic Expected-library publication."),
    # Tells the frontend this module needs a source folder of PDFs picked
    # before it can run (triggers the folder-picker popup).
    "source": "pdf_folder",
    "review_required": True,
}

from paths import KB_DIR, ROOT, SOURCES_MANIFEST
from pageindex.library import (
    SourceAssetCandidate,
    SourceCandidate,
    read_source_candidates,
    write_source_candidates,
)

OUT_DIR = ROOT / ".betteringest_out"        # BetterIngest working area (crops, raw md)
OCR_CACHE_DIR = ROOT / ".ocr_cache"         # BetterIngester's own OCR cache format
CONFIG_PATH = ROOT / ".betteringest.json"


# ── source-folder config ─────────────────────────────────────────────────────

def get_source_dir() -> str | None:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("source_dir")
    except Exception:
        return None


def set_source_dir(path: str) -> None:
    CONFIG_PATH.write_text(json.dumps({"source_dir": str(path)}, indent=2),
                           encoding="utf-8")


def _load_sources() -> dict[str, SourceCandidate]:
    return read_source_candidates(SOURCES_MANIFEST)


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


def prepare_review(source_dir: str | None = None, progress=None):
    """Reconstruct a folder into private candidates without publishing it."""
    src = Path(source_dir or get_source_dir() or "")
    if not src or not src.is_dir():
        raise FileNotFoundError(
            f"BetterIngest source folder not set or missing: '{src}'."
        )
    pdfs = sorted(src.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {src}/")
    _ingester, documents = _prepare_pdfs(pdfs, progress)
    return documents


def prepare_review_paths(paths: list[str], progress=None):
    """Reconstruct explicit PDFs into one unpublished review batch."""
    pdfs: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdfs.append(path)
        elif path.is_dir():
            pdfs.extend(sorted(path.glob("*.pdf")))
        else:
            raise FileNotFoundError(f"Not a PDF file or folder: {path}")
    if not pdfs:
        raise FileNotFoundError("No PDF files found in the given path(s).")
    _ingester, documents = _prepare_pdfs(pdfs, progress)
    return documents


def _require_ocr_dependencies() -> None:
    try:
        import paddle  # noqa: F401
        import paddlex  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "betteringest_pdf needs the locked OCR dependency group, missing "
            f"from this Python ({sys.executable}): {exc}. Run `uv sync --group ocr`."
        ) from exc


def _prepare_pdfs(pdfs: list[Path], progress=None):
    """Run deterministic reconstruction and return unpublished documents."""
    _require_ocr_dependencies()
    from modules.ingest._betteringest import BetterIngest

    ingester = BetterIngest(out_dir=OUT_DIR, cache_dir=OCR_CACHE_DIR)
    documents = []
    for index, pdf in enumerate(pdfs):
        if progress:
            progress({
                "phase": "ocr",
                "doc": pdf.stem,
                "done": index,
                "total": len(pdfs),
                "message": f"Reconstructing {pdf.name} (layout OCR + hierarchy)…",
            })
        documents.append(ingester.ingest(pdf))
    return ingester, documents


def _ingest_pdfs(pdfs: list[Path], progress=None,
                 caption_backend: str | None = None,
                 caption_model: str | None = None) -> dict:
    import os

    from modules.ingest._captioning import CaptioningUnavailable, caption_assets
    from modules.ingest._massage import massage

    backend = caption_backend or os.environ.get("BETTERINGEST_CAPTION_BACKEND",
                                                "ollama")
    KB_DIR.mkdir(exist_ok=True)
    bi, documents = _prepare_pdfs(pdfs, progress)
    sources = _load_sources()
    warnings: list[str] = []
    done_stems: list[str] = []

    def report(phase: str, doc: str, i: int, message: str = ""):
        info = {"phase": phase, "doc": doc, "done": i, "total": len(pdfs),
                "message": message}
        if progress:
            progress(info)

    for i, (pdf, doc) in enumerate(zip(pdfs, documents)):
        stem = pdf.stem
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

        sources[stem] = SourceCandidate(
            pdf_path=pdf.resolve(),
            ocr_scale=doc.ocr_scale,
            assets=tuple(SourceAssetCandidate(
                asset_id=asset.asset_id,
                path=Path(asset.image),
                media_type=(
                    mimetypes.guess_type(asset.image)[0]
                    or "application/octet-stream"
                ),
            ) for asset in doc.assets),
        )
        write_source_candidates(SOURCES_MANIFEST, sources)
        done_stems.append(stem)
        print(f"  {pdf.name} → {KB_DIR / (stem + '.md')} "
              f"({len(doc.assets)} assets)")
        report("done-doc", stem, i + 1)

    print(f"\nIngested {len(done_stems)} PDF(s) into {KB_DIR}/")
    return {"docs": done_stems, "warnings": warnings}
