"""Rendering a source page with its cited passage highlighted.

No test reached this route before, so a refactor could leave it raising
NameError on every request with the whole suite green.
"""


import pytest
from fastapi.testclient import TestClient


import server
from api import pdf
from api.routers import documents as documents_router

pdfium = pytest.importorskip("pypdfium2")


@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A two-page PDF registered in the manifest, as PDF ingest would leave it."""
    path = tmp_path / "guide.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 400)
    doc.new_page(200, 400)
    doc.save(str(path))
    monkeypatch.setattr(
        documents_router, "load_sources",
        lambda: {"guide": {"pdf": str(path), "ocr_scale": 2.0}},
    )
    pdf._png_cache.clear()
    return path


class TestRenderedPageSize:
    """The denominator of the whole provenance chain.

    A pin's bbox is in render pixels at the document's ingest ocr_scale, and
    _normalized_bbox divides by this to get the 0..1 fractions the client
    overlays. Get it wrong and every highlight is misplaced by that ratio — with
    no error anywhere, because the numbers stay perfectly plausible.
    """

    @pytest.fixture
    def sized_pdf(self, tmp_path, monkeypatch):
        path = tmp_path / "guide.pdf"
        doc = pdfium.PdfDocument.new()
        doc.new_page(200, 400)
        doc.new_page(200, 400)
        doc.save(str(path))
        monkeypatch.setattr(
            pdf, "load_sources",
            lambda: {"guide": {"pdf": str(path), "ocr_scale": 2.0}},
        )
        pdf._page_size_cache.clear()
        return path

    def test_the_page_size_is_scaled_by_the_ingest_scale(self, sized_pdf):
        # 200x400pt at ocr_scale 2.0 is what the pin's pixels were measured in.
        assert pdf._rendered_page_size("guide", 1) == (400.0, 800.0)

    def test_an_unknown_document_has_no_size(self, sized_pdf):
        assert pdf._rendered_page_size("never-ingested", 1) is None

    def test_a_page_outside_the_document_has_no_size(self, sized_pdf):
        assert pdf._rendered_page_size("guide", 3) is None
        assert pdf._rendered_page_size("guide", 0) is None

    def test_a_deleted_source_pdf_has_no_size(self, sized_pdf):
        sized_pdf.unlink()
        assert pdf._rendered_page_size("guide", 1) is None

    def test_the_size_is_cached(self, sized_pdf):
        pdf._rendered_page_size("guide", 1)
        assert len(pdf._page_size_cache) == 1

    def test_reingesting_the_pdf_invalidates_the_cached_size(self, sized_pdf, monkeypatch):
        """Keyed by mtime: a re-ingest that changes the page geometry must not
        keep normalizing new pins against the old page."""
        assert pdf._rendered_page_size("guide", 1) == (400.0, 800.0)
        doc = pdfium.PdfDocument.new()
        doc.new_page(100, 100)
        doc.save(str(sized_pdf))
        import os, time
        os.utime(sized_pdf, (time.time() + 10, time.time() + 10))
        assert pdf._rendered_page_size("guide", 1) == (200.0, 200.0)

    def test_the_cache_is_bounded(self, tmp_path, monkeypatch):
        """Needs more distinct pages than the cap: asking for two pages twice
        would sit inside the cap whether eviction worked or not."""
        path = tmp_path / "long.pdf"
        doc = pdfium.PdfDocument.new()
        for _ in range(6):
            doc.new_page(200, 400)
        doc.save(str(path))
        monkeypatch.setattr(
            pdf, "load_sources", lambda: {"long": {"pdf": str(path), "ocr_scale": 1.0}})
        pdf._page_size_cache.clear()
        monkeypatch.setattr(pdf, "_PAGE_SIZE_CACHE_MAX", 2)

        for page in range(1, 7):
            assert pdf._rendered_page_size("long", page) == (200.0, 400.0)
        assert len(pdf._page_size_cache) <= 2, "six distinct pages must not all be retained"


class TestPageRender:
    def test_renders_a_page_as_png(self, client, corpus):
        response = client.get("/api/document/guide/page/1")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_renders_with_a_highlighted_bbox(self, client, corpus):
        response = client.get("/api/document/guide/page/1", params={"bbox": "10,20,150,60"})
        assert response.status_code == 200
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_a_malformed_bbox_still_renders_the_page(self, client, corpus):
        # The highlight is an overlay; losing it must not lose the page.
        response = client.get("/api/document/guide/page/1", params={"bbox": "not,a,bbox,at all"})
        assert response.status_code == 200

    def test_page_out_of_range_is_404(self, client, corpus):
        assert client.get("/api/document/guide/page/9").status_code == 404

    def test_unknown_document_is_404(self, client, corpus):
        assert client.get("/api/document/nope/page/1").status_code == 404

    @pytest.mark.parametrize("stem", ["../../etc/passwd", "..%2f..%2fsecret"])
    def test_path_traversal_is_rejected(self, client, corpus, stem):
        response = client.get(f"/api/document/{stem}/page/1")
        assert response.status_code == 404
        assert "passwd" not in response.text


class TestPngCache:
    def test_a_repeated_request_is_served_from_cache(self, client, corpus):
        first = client.get("/api/document/guide/page/1")
        assert len(pdf._png_cache) == 1
        second = client.get("/api/document/guide/page/1")
        assert second.content == first.content
        assert len(pdf._png_cache) == 1

    def test_a_different_bbox_is_a_different_entry(self, client, corpus):
        client.get("/api/document/guide/page/1", params={"bbox": "1,2,3,4"})
        client.get("/api/document/guide/page/1", params={"bbox": "5,6,7,8"})
        assert len(pdf._png_cache) == 2

    def test_the_cache_is_bounded(self, corpus):
        for i in range(pdf._PNG_CACHE_MAX + 10):
            pdf.cache_png(("k", i), b"x")
        assert len(pdf._png_cache) <= pdf._PNG_CACHE_MAX
