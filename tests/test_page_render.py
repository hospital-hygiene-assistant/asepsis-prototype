"""Rendering an immutable source page with its cited passage highlighted."""

import hashlib
import pytest
from fastapi.testclient import TestClient


import server
from api import pdf
from api.routers import documents as documents_router
from pageindex.library import (
    ExpectedLibraryStore,
    LibraryCandidate,
    SourceCandidate,
    SourceDocument,
)

pdfium = pytest.importorskip("pypdfium2")


@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A two-page PDF published into the current immutable library."""
    path = tmp_path / "guide.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 400)
    doc.new_page(200, 400)
    doc.save(str(path))
    library = tmp_path / "library"
    snapshot = ExpectedLibraryStore(library).publish({
        "guide": LibraryCandidate(
            "# Guide\n\nSource passage.", SourceCandidate(path, 2.0)
        )
    })
    monkeypatch.setattr(documents_router, "LIBRARY_DIR", library)
    pdf._png_cache.clear()
    return f"/api/library/{snapshot.generation_id}/documents/guide"


def source_document(path, scale=2.0):
    return SourceDocument(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        _pdf_path=path,
        ocr_scale=scale,
        href="/immutable/source.pdf",
    )


class TestRenderedPageSize:
    """The generation-bound denominator of the provenance chain."""

    @pytest.fixture
    def sized_pdf(self, tmp_path):
        path = tmp_path / "guide.pdf"
        doc = pdfium.PdfDocument.new()
        doc.new_page(200, 400)
        doc.new_page(200, 400)
        doc.save(str(path))
        pdf._page_size_cache.clear()
        return source_document(path)

    def test_the_page_size_is_scaled_by_the_ingest_scale(self, sized_pdf):
        # 200x400pt at ocr_scale 2.0 is what the pin's pixels were measured in.
        assert pdf.rendered_page_size(
            sized_pdf, 1, pin_scale=2.0
        ) == (400.0, 800.0)

    def test_a_different_pin_scale_has_no_size(self, sized_pdf):
        assert pdf.rendered_page_size(sized_pdf, 1, pin_scale=4.0) is None

    def test_a_page_outside_the_document_has_no_size(self, sized_pdf):
        assert pdf.rendered_page_size(sized_pdf, 3, pin_scale=2.0) is None
        assert pdf.rendered_page_size(sized_pdf, 0, pin_scale=2.0) is None

    def test_a_deleted_source_pdf_has_no_size(self, sized_pdf):
        sized_pdf._pdf_path.unlink()
        assert pdf.rendered_page_size(sized_pdf, 1, pin_scale=2.0) is None

    def test_the_size_is_cached(self, sized_pdf):
        pdf.rendered_page_size(sized_pdf, 1, pin_scale=2.0)
        assert len(pdf._page_size_cache) == 1

    def test_replacing_the_bound_pdf_is_rejected(self, sized_pdf):
        assert pdf.rendered_page_size(
            sized_pdf, 1, pin_scale=2.0
        ) == (400.0, 800.0)
        doc = pdfium.PdfDocument.new()
        doc.new_page(100, 100)
        doc.save(str(sized_pdf._pdf_path))

        assert pdf.rendered_page_size(sized_pdf, 1, pin_scale=2.0) is None

    def test_the_cache_is_bounded(self, tmp_path, monkeypatch):
        """Needs more distinct pages than the cap: asking for two pages twice
        would sit inside the cap whether eviction worked or not."""
        path = tmp_path / "long.pdf"
        doc = pdfium.PdfDocument.new()
        for _ in range(6):
            doc.new_page(200, 400)
        doc.save(str(path))
        source = source_document(path, scale=1.0)
        pdf._page_size_cache.clear()
        monkeypatch.setattr(pdf, "_PAGE_SIZE_CACHE_MAX", 2)

        for page in range(1, 7):
            assert pdf.rendered_page_size(
                source, page, pin_scale=1.0
            ) == (200.0, 400.0)
        assert len(pdf._page_size_cache) <= 2, "six distinct pages must not all be retained"


class TestPageRender:
    def test_renders_a_page_as_png(self, client, corpus):
        response = client.get(f"{corpus}/page/1")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_renders_with_a_highlighted_bbox(self, client, corpus):
        response = client.get(f"{corpus}/page/1", params={"bbox": "10,20,150,60"})
        assert response.status_code == 200
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_a_malformed_bbox_still_renders_the_page(self, client, corpus):
        # The highlight is an overlay; losing it must not lose the page.
        response = client.get(f"{corpus}/page/1", params={"bbox": "not,a,bbox,at all"})
        assert response.status_code == 200

    def test_page_out_of_range_is_404(self, client, corpus):
        assert client.get(f"{corpus}/page/9").status_code == 404

    def test_unknown_document_is_404(self, client, corpus):
        assert client.get(f"{corpus.replace('/guide', '/nope')}/page/1").status_code == 404

    @pytest.mark.parametrize("stem", ["../../etc/passwd", "..%2f..%2fsecret"])
    def test_path_traversal_is_rejected(self, client, corpus, stem):
        prefix = corpus.rsplit("/", 1)[0]
        response = client.get(f"{prefix}/{stem}/page/1")
        assert response.status_code == 404
        assert "passwd" not in response.text


class TestPngCache:
    def test_a_repeated_request_is_served_from_cache(self, client, corpus):
        first = client.get(f"{corpus}/page/1")
        assert len(pdf._png_cache) == 1
        second = client.get(f"{corpus}/page/1")
        assert second.content == first.content
        assert len(pdf._png_cache) == 1

    def test_a_different_bbox_is_a_different_entry(self, client, corpus):
        client.get(f"{corpus}/page/1", params={"bbox": "1,2,3,4"})
        client.get(f"{corpus}/page/1", params={"bbox": "5,6,7,8"})
        assert len(pdf._png_cache) == 2

    def test_the_cache_is_bounded(self, corpus):
        for i in range(pdf._PNG_CACHE_MAX + 10):
            pdf.cache_png(("k", i), b"x")
        assert len(pdf._png_cache) <= pdf._PNG_CACHE_MAX
