"""The ingested corpus: trees, markdown, source PDFs and rendered pages."""

import json
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, Response

from pageindex.library import (
    ExpectedLibraryCorrupt,
    ExpectedLibraryNotBuilt,
    ExpectedLibraryStore,
    validate_library_segment,
)
from paths import LIBRARY_DIR

from ..pdf import cache_png, cached_png
from ..trees import leaf_count, read_tree

router = APIRouter()


def _generation_document(generation_id: str, document_id: str):
    try:
        validate_library_segment(document_id, "document identity")
        return ExpectedLibraryStore(LIBRARY_DIR).open_generation(
            generation_id
        ).document(document_id)
    except (KeyError, ValueError):
        return JSONResponse({"error": "document not found"}, status_code=404)
    except ExpectedLibraryCorrupt:
        return JSONResponse(
            {"error": "Expected library generation is unavailable"},
            status_code=503,
        )


@router.get("/api/library/{generation_id}/documents/{document_id}/pdf")
def get_generation_document_pdf(generation_id: str, document_id: str):
    """Serve the exact immutable PDF bound to one Expected library generation."""
    document = _generation_document(generation_id, document_id)
    if isinstance(document, JSONResponse):
        return document
    if document.source is None:
        return JSONResponse({"error": "document has no source PDF"}, status_code=404)
    return FileResponse(
        document.source.pdf_path,
        media_type="application/pdf",
        filename=f"{document.document_id}.pdf",
    )


@router.get("/api/library/{generation_id}/documents/{document_id}/full")
def get_generation_document_full(generation_id: str, document_id: str):
    """Return canonical Markdown from the same immutable generation."""
    document = _generation_document(generation_id, document_id)
    if isinstance(document, JSONResponse):
        return document
    markdown = document.canonical_markdown
    if document.source is not None:
        for asset in document.source.assets:
            markdown = markdown.replace(
                f"/assets/{document.document_id}/{asset.filename}", asset.href
            )
    return JSONResponse({
        "document_id": document.document_id,
        "generation_id": generation_id,
        "markdown": markdown,
    })


@router.get(
    "/api/library/{generation_id}/documents/{document_id}/assets/{asset_id}"
)
def get_generation_document_asset(
    generation_id: str, document_id: str, asset_id: str
):
    """Serve one immutable provenance asset from the bound generation."""
    document = _generation_document(generation_id, document_id)
    if isinstance(document, JSONResponse):
        return document
    if document.source is None:
        return JSONResponse({"error": "source asset not found"}, status_code=404)
    asset = next(
        (item for item in document.source.assets if item.asset_id == asset_id),
        None,
    )
    if asset is None:
        return JSONResponse({"error": "source asset not found"}, status_code=404)
    return FileResponse(asset.path, media_type=asset.media_type)


@router.get("/api/documents")
def get_documents():
    docs = []
    try:
        snapshot = ExpectedLibraryStore(LIBRARY_DIR).open_current()
        for document in snapshot.documents:
            tree = read_tree(document.index_path)
            base_href = (
                f"/api/library/{snapshot.generation_id}/documents/"
                f"{quote(document.document_id, safe='')}"
            )
            docs.append({
                "name": document.document_id,
                "generation_id": snapshot.generation_id,
                "leaf_count": leaf_count(tree),
                "tree": tree,
                "full_href": f"{base_href}/full",
                "source_href": (
                    document.source.href if document.source is not None else None
                ),
                "page_href": f"{base_href}/page/{{page}}",
            })
    except (ExpectedLibraryNotBuilt, ExpectedLibraryCorrupt):
        return JSONResponse(
            {"error": "Expected library is unavailable"}, status_code=503
        )
    return JSONResponse(docs)


@router.get(
    "/api/library/{generation_id}/documents/{document_id}/page/{page}"
)
def get_generation_document_page(
    generation_id: str,
    document_id: str,
    page: int,
    bbox: Optional[str] = None,
    regions: Optional[str] = None,
):
    """Render one page from one immutable Expected library generation."""
    document = _generation_document(generation_id, document_id)
    if isinstance(document, JSONResponse):
        return document
    return _render_document_page(document, page, bbox, regions)


def _render_document_page(document, page, bbox, regions):
    if document.source is None:
        return JSONResponse({"error": "document has no source PDF"}, status_code=404)
    pdf_path = document.source.pdf_path
    try:
        import pypdfium2 as pdfium
        from PIL import ImageDraw
    except ImportError as exc:
        return JSONResponse(
            {"error": f"page rendering needs pypdfium2 + Pillow: {exc}"},
            status_code=501)

    cache_key = (document.source.sha256, page, bbox or "", regions or "")
    cached = cached_png(cache_key)
    if cached is not None:
        return Response(content=cached, media_type="image/png")

    scale = document.source.ocr_scale
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
