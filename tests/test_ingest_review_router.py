"""The local review HTTP interface exposes no filesystem paths and fails closed."""

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server
from api.routers import reviews as reviews_router
from modules.ingest._betteringest.betteringest import Asset, IngestedDoc
from modules.ingest._betteringest.ocr import Block
from modules.ingest.review import ReviewStore

pdfium = pytest.importorskip("pypdfium2")


def _edit(region):
    return {
        key: value
        for key, value in region.model_dump(mode="json").items()
        if key in {
            "region_id", "kind", "page", "box", "caption",
            "asset_filename", "deleted",
        }
    }


@pytest.fixture
def review_client(tmp_path, monkeypatch):
    pdf = tmp_path / "incoming.pdf"
    source = pdfium.PdfDocument.new()
    source.new_page(200, 300)
    source.save(str(pdf))
    crop = tmp_path / "figure_1.png"
    Image.new("RGB", (50, 50), "white").save(crop)
    document = IngestedDoc(
        doc_name="guide",
        title="Guide",
        pdf_path=str(pdf),
        md_path="",
        markdown="# Guide\n\nVerified body.\n\n![figure 1](assets/figure_1.png)",
        assets=[Asset(
            "figure_1", "figure", 1, "Figure 1: Example", 1, str(crop),
            bbox=[10, 20, 110, 120],
        )],
        tree=None,
        blocks=[Block("doc_title", "Guide", 0, (10, 10, 100, 20))],
        ladder_diag={},
        ocr_scale=2.0,
    )
    store = ReviewStore(tmp_path / "reviews", tmp_path / "library")
    session = store.create([document])
    monkeypatch.setattr(reviews_router, "review_store", store)
    return TestClient(server.app), session


def test_session_view_exposes_opaque_links_not_paths(review_client):
    client, session = review_client
    response = client.get(f"/api/ingest/reviews/{session.session_id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session.session_id
    assert "pdf_filename" not in payload["documents"][0]
    assert "markdown" not in payload["documents"][0]
    assert "/tmp/" not in response.text


def test_review_page_is_session_scoped_png(review_client):
    client, session = review_client
    response = client.get(
        f"/api/ingest/reviews/{session.session_id}/documents/guide/pages/1.png"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_unknown_fields_are_rejected_before_mutation(review_client):
    client, session = review_client
    region = session.documents[0].regions[0].model_dump(mode="json")
    response = client.put(
        f"/api/ingest/reviews/{session.session_id}/documents/guide/regions",
        json={
            "expected_revision": session.revision,
            "regions": [region],
            "unexpected": True,
        },
    )
    assert response.status_code == 422


def test_stale_revision_is_structured_conflict(review_client):
    client, session = review_client
    region = _edit(session.documents[0].regions[0])
    path = f"/api/ingest/reviews/{session.session_id}/documents/guide/regions"
    assert client.put(path, json={
        "expected_revision": session.revision, "regions": [region]
    }).status_code == 200
    stale = client.put(path, json={
        "expected_revision": session.revision, "regions": [region]
    })
    assert stale.status_code == 409
    assert stale.json()["code"] == "review_conflict"


def test_table_outcomes_cannot_be_forged_through_geometry_updates(review_client):
    client, session = review_client
    region = _edit(session.documents[0].regions[0])
    region.update({
        "kind": "table",
        "table_state": "recognized",
        "table_markdown": "| invented |",
    })

    response = client.put(
        f"/api/ingest/reviews/{session.session_id}/documents/guide/regions",
        json={"expected_revision": session.revision, "regions": [region]},
    )

    assert response.status_code == 422


def test_confirm_returns_the_immutable_generation(review_client):
    client, session = review_client
    response = client.post(
        f"/api/ingest/reviews/{session.session_id}/confirm",
        json={"expected_revision": session.revision},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "published"
    assert len(payload["published_generation_id"]) == 64


def test_console_security_headers_and_vendored_sanitizer(review_client):
    client, _session = review_client
    root = client.get("/")
    assert root.status_code == 200
    assert "object-src 'none'" in root.headers["content-security-policy"]
    assert "/static/dompurify.min.js" in root.text
    sanitizer = client.get("/static/dompurify.min.js")
    assert sanitizer.status_code == 200
    assert "DOMPurify" in sanitizer.text
