"""Source-PDF geometry.

A pin records its bbox in render pixels at the document's ingest ocr_scale. Only
the server knows that scale and the page size, so turning a pin into page
fractions a client can overlay has to happen here.
"""

import threading
from pathlib import Path
from typing import Optional

from .sources import load_sources

# Page sizes are stable per (pdf, mtime, page) and cost a pdfium open to read,
# so memoize them off the chat hot path.
_page_size_cache: dict = {}
_page_size_lock = threading.Lock()
_PAGE_SIZE_CACHE_MAX = 256


def _rendered_page_size(stem: str, page: int) -> Optional[tuple]:
    """(width, height) of one PDF page in render pixels at its ingest ocr_scale.

    Pin bboxes are expressed in exactly these coordinates, so this is the
    denominator that normalizes them.
    """
    src = load_sources().get(stem)
    if not src:
        return None
    pdf_path = Path(src["pdf"])
    if not pdf_path.exists():
        return None
    try:
        key = (str(pdf_path), pdf_path.stat().st_mtime, page)
    except OSError:
        return None
    with _page_size_lock:
        hit = _page_size_cache.get(key)
    if hit is not None:
        return hit
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    try:
        doc = pdfium.PdfDocument(str(pdf_path))
        if not (1 <= page <= len(doc)):
            return None
        scale = float(src.get("ocr_scale", 2.0))
        width_pt, height_pt = doc[page - 1].get_size()
        size = (width_pt * scale, height_pt * scale)
    except Exception:
        return None
    with _page_size_lock:
        if len(_page_size_cache) >= _PAGE_SIZE_CACHE_MAX:
            _page_size_cache.pop(next(iter(_page_size_cache)))
        _page_size_cache[key] = size
    return size


def _normalized_bbox(stem: str, pin: dict) -> Optional[dict]:
    """A pin bbox as 0..1 fractions of the page: {page, x, y, width, height}.

    Pins carry corners in render pixels at ocr_scale. Only the server knows that
    scale and the page size, so a client cannot do this conversion itself.
    Origin is top-left on both sides, matching how the pins were produced.
    """
    bbox = pin.get("bbox")
    page = pin.get("page")
    if not page or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    size = _rendered_page_size(stem, int(page))
    if not size:
        return None
    page_w, page_h = size
    if page_w <= 0 or page_h <= 0:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None

    def clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    # Clamp both corners, then derive the extents from the clamped corners.
    # Clamping the origin but sizing from the raw span would keep the full width
    # of a box whose left edge was off-page, drawing the highlight past the
    # passage and over unrelated text.
    left, right = clamp(x0 / page_w), clamp(x1 / page_w)
    top, bottom = clamp(y0 / page_h), clamp(y1 / page_h)
    width, height = right - left, bottom - top
    # Zero-extent, inverted, or fully off-page boxes have no honest rendering,
    # and the consuming schema requires positive extents.
    if width <= 0 or height <= 0:
        return None
    return {"page": int(page), "x": left, "y": top, "width": width, "height": height}
