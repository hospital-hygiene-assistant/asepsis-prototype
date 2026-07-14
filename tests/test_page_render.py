"""Rendering a source page with its cited passage highlighted.

No test reached this route before, so a refactor could leave it raising
NameError on every request with the whole suite green.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tauri-app"))

import server  # noqa: E402
from api import pdf  # noqa: E402
from api.routers import documents as documents_router  # noqa: E402

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
