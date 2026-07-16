"""Source-PDF geometry.

A pin records its bbox in render pixels at the document's ingest ocr_scale. Only
the server knows that scale and the page size, so turning a pin into page
fractions a client can overlay has to happen here.
"""

import hashlib
import threading
from typing import Optional

from pageindex.library import SourceDocument

# Page sizes are stable per (source digest, page, ingest scale) and cost a
# pdfium open to read, so memoize them off the chat hot path.
_page_size_cache: dict = {}
_page_size_lock = threading.Lock()
_PAGE_SIZE_CACHE_MAX = 256


def rendered_page_size(
    source: SourceDocument, page: int, *, pin_scale: float
) -> Optional[tuple[float, float]]:
    """Rendered size from one generation-bound source and its own pin scale."""
    if pin_scale != source.ocr_scale or page < 1:
        return None
    try:
        if hashlib.sha256(source.pdf_path.read_bytes()).hexdigest() != source.sha256:
            return None
    except OSError:
        return None
    key = (source.sha256, page, pin_scale)
    with _page_size_lock:
        hit = _page_size_cache.get(key)
    if hit is not None:
        return hit
    try:
        import pypdfium2 as pdfium
        document = pdfium.PdfDocument(str(source.pdf_path))
        if page > len(document):
            return None
        width_pt, height_pt = document[page - 1].get_size()
        size = (width_pt * pin_scale, height_pt * pin_scale)
    except Exception:
        return None
    with _page_size_lock:
        if len(_page_size_cache) >= _PAGE_SIZE_CACHE_MAX:
            _page_size_cache.pop(next(iter(_page_size_cache)))
        _page_size_cache[key] = size
    return size


# Chat citation previews request the same page and bbox repeatedly, and a pdfium
# render is ~100ms, so memoize the PNG bytes.
_png_cache: dict = {}
_png_lock = threading.Lock()
_PNG_CACHE_MAX = 64


def cached_png(key: tuple):
    with _png_lock:
        return _png_cache.get(key)


def cache_png(key: tuple, data: bytes) -> None:
    with _png_lock:
        if len(_png_cache) >= _PNG_CACHE_MAX:
            _png_cache.pop(next(iter(_png_cache)))
        _png_cache[key] = data
