"""The ingested corpus: trees, markdown, source PDFs and rendered pages."""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, Response

from paths import INDEX_DIR, KB_DIR

from ..pdf import cache_png, cached_png
from ..sources import load_sources
from ..trees import leaf_count, read_tree

router = APIRouter()


@router.get("/api/documents")
def get_documents():
    docs = []
    for idx in sorted(INDEX_DIR.glob("*.json")):
        tree = read_tree(idx)
        docs.append({
            "name": idx.stem,
            "leaf_count": leaf_count(tree),
            "tree": tree,
        })
    return JSONResponse(docs)


@router.get("/api/document/{stem}/full")
def get_document_full(stem: str):
    """Return the raw markdown for a single knowledge-base document.

    The frontend renders the markdown and derives its own table of contents
    from the rendered headings, so anchors always stay consistent.
    """

    # Guard against path traversal — only allow plain stems that exist.
    safe_stem = Path(stem).name
    md_path = KB_DIR / f"{safe_stem}.md"
    if not md_path.exists() or md_path.parent.resolve() != KB_DIR.resolve():
        return JSONResponse({"error": f"Document '{stem}' not found"}, status_code=404)

    return JSONResponse({
        "stem": safe_stem,
        "markdown": md_path.read_text(encoding="utf-8"),
    })


@router.get("/api/document/{stem}/pdf")
def get_document_pdf(stem: str):
    """Serve a document's source PDF so a client can render it itself.

    The path is resolved from the server-side manifest keyed by a bare stem, so
    `stem` never reaches the filesystem — same guard as the other document routes.
    """
    safe_stem = Path(stem).name
    src = load_sources().get(safe_stem)
    if not src:
        return JSONResponse(
            {"error": f"'{safe_stem}' has no source PDF (not ingested from a PDF)"},
            status_code=404)
    pdf_path = Path(src["pdf"])
    if not pdf_path.exists():
        return JSONResponse({"error": f"source PDF for '{safe_stem}' is unavailable"},
                            status_code=404)
    return FileResponse(pdf_path, media_type="application/pdf",
                        filename=f"{safe_stem}.pdf")


@router.get("/api/document/{stem}/page/{page}")
def get_document_page(stem: str, page: int, bbox: Optional[str] = None,
                      regions: Optional[str] = None):
    """Render one page of the document's SOURCE PDF as a PNG, optionally with
    the pin's bbox highlighted — powers 'jump to its page/bbox' in the UI.

    `bbox` is "x0,y0,x1,y1" and `regions` is JSON [[page,x0,y0,x1,y1],...],
    both in render pixels at the ingest ocr_scale (the pin convention)."""
    src = load_sources().get(Path(stem).name)
    if not src:
        return JSONResponse(
            {"error": f"'{stem}' has no source PDF (not ingested from a PDF)"},
            status_code=404)
    pdf_path = Path(src["pdf"])
    if not pdf_path.exists():
        return JSONResponse({"error": f"source PDF moved or deleted: {pdf_path}"},
                            status_code=404)
    try:
        import pypdfium2 as pdfium
        from PIL import ImageDraw
    except ImportError as exc:
        return JSONResponse(
            {"error": f"page rendering needs pypdfium2 + Pillow: {exc}"},
            status_code=501)

    cache_key = (str(pdf_path), pdf_path.stat().st_mtime, page, bbox or "", regions or "")
    cached = cached_png(cache_key)
    if cached is not None:
        return Response(content=cached, media_type="image/png")

    scale = float(src.get("ocr_scale", 2.0))
    doc = pdfium.PdfDocument(str(pdf_path))
    if not (1 <= page <= len(doc)):
        return JSONResponse({"error": f"page {page} out of range 1..{len(doc)}"},
                            status_code=404)
    img = doc[page - 1].render(scale=scale).to_pil().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    if regions:
        try:
            for r in json.loads(regions):
                if int(r[0]) == page:
                    draw.rectangle([r[1], r[2], r[3], r[4]],
                                   outline=(56, 189, 248, 220), width=3)
        except Exception:
            pass
    if bbox:
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox.split(","))
            draw.rectangle([x0, y0, x1, y1], fill=(245, 158, 11, 56),
                           outline=(245, 158, 11, 255), width=4)
        except Exception:
            pass

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    cache_png(cache_key, data)
    return Response(content=data, media_type="image/png")
