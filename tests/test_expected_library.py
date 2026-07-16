"""The Expected library binds searchable text to immutable source evidence."""

import hashlib
import json

from pageindex.library import (
    ExpectedLibraryCorrupt,
    ExpectedLibraryNotBuilt,
    ExpectedLibraryStore,
    LibraryCandidate,
    SourceAssetCandidate,
    SourceCandidate,
    read_source_candidates,
    write_source_candidates,
)
import pytest
from pageindex.nodes import PageNode
from pageindex.pins import (
    AssetProvenance,
    PixelBox,
    ProvenancePin,
    SourceSpan,
    emit_pin,
)
from pageindex.serialization import deserialize_document, serialize_document


def test_published_source_pdf_is_immutable_and_generation_scoped(tmp_path):
    source_pdf = tmp_path / "incoming" / "guide.pdf"
    source_pdf.parent.mkdir()
    original = b"%PDF-1.7\noriginal guideline\n"
    source_pdf.write_bytes(original)
    candidate = LibraryCandidate(
        canonical_markdown="# PPE\n\nWear gloves.\n",
        source=SourceCandidate(
            pdf_path=source_pdf,
            ocr_scale=2.0,
        ),
    )

    snapshot = ExpectedLibraryStore(tmp_path / "library").publish(
        {"guide": candidate}
    )
    source_pdf.write_bytes(b"%PDF-1.7\nreplacement\n")

    published = snapshot.document("guide").source
    assert published is not None
    assert published.read_pdf() == original
    assert published.sha256 == hashlib.sha256(original).hexdigest()
    assert published.href == (
        f"/api/library/{snapshot.generation_id}/documents/guide/pdf"
    )


def test_published_markdown_and_source_assets_are_immutable(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    pdf = incoming / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.7\nsource\n")
    crop = incoming / "figure.png"
    crop.write_bytes(b"\x89PNG\r\n\x1a\noriginal crop")
    candidate = LibraryCandidate(
        canonical_markdown="# Figure\n\nOriginal explanation.\n",
        source=SourceCandidate(
            pdf_path=pdf,
            ocr_scale=2.0,
            assets=(SourceAssetCandidate(
                asset_id="figure_1",
                path=crop,
                media_type="image/png",
            ),),
        ),
    )

    snapshot = ExpectedLibraryStore(tmp_path / "library").publish(
        {"guide": candidate}
    )
    crop.write_bytes(b"replacement crop")

    document = snapshot.document("guide")
    assert document.canonical_markdown == "# Figure\n\nOriginal explanation.\n"
    asset = document.source.assets[0]
    assert asset.read_bytes() == b"\x89PNG\r\n\x1a\noriginal crop"
    assert asset.href == (
        f"/api/library/{snapshot.generation_id}/documents/guide/"
        "assets/figure_1"
    )


def test_open_generation_rejects_a_missing_source_object(tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.7\nsource\n")
    store = ExpectedLibraryStore(tmp_path / "library")
    snapshot = store.publish({
        "guide": LibraryCandidate(
            canonical_markdown="# Guide\n\nSource text.\n",
            source=SourceCandidate(
                pdf_path=pdf,
                ocr_scale=2.0,
            ),
        ),
    })
    source = snapshot.document("guide").source
    (store.objects / "pdf" / f"{source.sha256}.pdf").unlink()

    with pytest.raises(ExpectedLibraryCorrupt, match="source PDF"):
        store.open_generation(snapshot.generation_id)


def test_open_generation_rejects_a_corrupt_source_asset(tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.7\nsource\n")
    asset = tmp_path / "figure.png"
    asset.write_bytes(b"original figure")
    store = ExpectedLibraryStore(tmp_path / "library")
    snapshot = store.publish({
        "guide": LibraryCandidate(
            canonical_markdown="# Guide\n\nSource text.\n",
            source=SourceCandidate(
                pdf_path=pdf,
                ocr_scale=2.0,
                assets=(SourceAssetCandidate(
                    asset_id="figure_1",
                    path=asset,
                    media_type="image/png",
                ),),
            ),
        ),
    })
    published_asset = snapshot.document("guide").source.assets[0]
    (store.objects / "assets" / published_asset.sha256).write_bytes(b"corrupt")

    with pytest.raises(ExpectedLibraryCorrupt, match="source asset"):
        store.open_generation(snapshot.generation_id)


def test_open_generation_rejects_a_replaced_valid_index(tmp_path):
    store = ExpectedLibraryStore(tmp_path / "library")
    snapshot = store.publish({
        "guide": LibraryCandidate(
            canonical_markdown="# Guide\n\nOriginal.\n",
        ),
    })
    replacement = serialize_document((PageNode(
        node_id="replacement",
        title="Replacement",
        heading_level=1,
        line_idx=0,
        summary="Different",
        content="Different.",
    ),))
    (store.generations / snapshot.generation_id / "guide.json").write_text(
        replacement, encoding="utf-8"
    )

    with pytest.raises(ExpectedLibraryCorrupt, match="index"):
        store.open_generation(snapshot.generation_id)


def test_publish_rejects_a_pin_scale_that_does_not_match_its_source(tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.7\nsource\n")
    pin = ProvenancePin(
        version=2,
        document="guide",
        node_id="guide",
        spans=(SourceSpan(1, 0, 12, PixelBox(10, 20, 100, 80)),),
        scale=2.0,
    )

    with pytest.raises(ValueError, match="pin scale"):
        ExpectedLibraryStore(tmp_path / "library").publish({
            "guide": LibraryCandidate(
                canonical_markdown=(
                    f"# Guide\n\n{emit_pin(pin)}\nSource text.\n"
                ),
                source=SourceCandidate(
                    pdf_path=pdf,
                    ocr_scale=4.0,
                ),
            ),
        })


def test_source_href_percent_encodes_the_generation_document_identity(tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.7\nsource\n")
    snapshot = ExpectedLibraryStore(tmp_path / "library").publish({
        "MRSA Leitlinie": LibraryCandidate(
            canonical_markdown="# Guide\n\nSource.\n",
            source=SourceCandidate(
                pdf_path=pdf,
                ocr_scale=2.0,
            ),
        ),
    })

    assert snapshot.documents[0].source.href.endswith(
        "/documents/MRSA%20Leitlinie/pdf"
    )


def test_publish_rejects_a_document_without_searchable_text(tmp_path):
    with pytest.raises(ValueError, match="searchable leaf"):
        ExpectedLibraryStore(tmp_path / "library").publish({
            "empty": LibraryCandidate(canonical_markdown="# Empty\n")
        })


def test_publish_rejects_an_asset_pin_without_an_immutable_source_asset(
    tmp_path,
):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.7\nsource\n")
    pin = ProvenancePin(
        version=2,
        document="guide",
        node_id="figure",
        spans=(SourceSpan(1, 0, 7, PixelBox(10, 20, 100, 80)),),
        scale=2.0,
        asset=AssetProvenance("figure_1", "figure", "/assets/figure.png"),
    )

    with pytest.raises(ValueError, match="undeclared source asset"):
        ExpectedLibraryStore(tmp_path / "library").publish({
            "guide": LibraryCandidate(
                canonical_markdown=f"# Figure\n\n{emit_pin(pin)}\nCaption",
                source=SourceCandidate(pdf, ocr_scale=2.0),
            )
        })


def test_source_pdf_requires_searchable_body_guidance_before_promotion(tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.7\nsource\n")
    asset = tmp_path / "figure.png"
    asset.write_bytes(b"figure")
    store = ExpectedLibraryStore(tmp_path / "library")
    current = store.publish({
        "stable": LibraryCandidate("# Stable\n\nSearchable body guidance.\n")
    })
    pin = ProvenancePin(
        version=2,
        document="image-only",
        node_id="figure",
        spans=(SourceSpan(1, 0, 7, PixelBox(10, 20, 100, 80)),),
        scale=2.0,
        asset=AssetProvenance("figure_1", "figure", "/assets/figure.png"),
    )

    with pytest.raises(ValueError, match="no searchable body guidance"):
        store.publish({
            "image-only": LibraryCandidate(
                canonical_markdown=f"# Figure\n\n{emit_pin(pin)}\nCaption",
                source=SourceCandidate(
                    pdf,
                    ocr_scale=2.0,
                    assets=(SourceAssetCandidate(
                        "figure_1", asset, "image/png"
                    ),),
                ),
            )
        })

    assert store.open_current().generation_id == current.generation_id


def test_open_generation_normalizes_a_missing_manifest(tmp_path):
    store = ExpectedLibraryStore(tmp_path / "library")

    with pytest.raises(ExpectedLibraryCorrupt, match="manifest"):
        store.open_generation("0" * 64)


def test_open_current_normalizes_a_malformed_pointer(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    (root / "current.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ExpectedLibraryCorrupt, match="current pointer"):
        ExpectedLibraryStore(root).open_current()


def test_open_current_does_not_treat_a_flat_index_as_a_library(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    (root / "guide.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ExpectedLibraryNotBuilt):
        ExpectedLibraryStore(root).open_current()


def test_generation_identity_rejects_a_rewritten_index_and_manifest(tmp_path):
    store = ExpectedLibraryStore(tmp_path / "library")
    snapshot = store.publish({
        "guide": LibraryCandidate(
            canonical_markdown="# Guide\n\nOriginal.\n",
        ),
    })
    replacement = serialize_document((PageNode(
        node_id="replacement",
        title="Replacement",
        heading_level=1,
        line_idx=0,
        summary="Different",
        content="Different.",
    ),))
    index_path = store.generations / snapshot.generation_id / "guide.json"
    index_path.write_text(replacement, encoding="utf-8")
    manifest_path = index_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][0]["index_sha256"] = hashlib.sha256(
        replacement.encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ExpectedLibraryCorrupt, match="generation identity"):
        store.open_generation(snapshot.generation_id)


def test_publish_does_not_promote_a_corrupt_existing_generation(tmp_path):
    store = ExpectedLibraryStore(tmp_path / "library")
    first_candidate = LibraryCandidate("# First\n\nStable content.\n")
    second_candidate = LibraryCandidate("# Second\n\nOther content.\n")
    first = store.publish({"first": first_candidate})
    second = store.publish({"second": second_candidate})
    store.publish({"first": first_candidate})
    (store.generations / second.generation_id / "second.json").write_text(
        "[]", encoding="utf-8"
    )

    with pytest.raises(ExpectedLibraryCorrupt, match="document index"):
        store.publish({"second": second_candidate})

    assert store.open_current().generation_id == first.generation_id


def test_generation_scoped_document_adapters_serve_one_immutable_bundle(
    tmp_path, monkeypatch
):
    pdfium = pytest.importorskip("pypdfium2")
    from fastapi.testclient import TestClient
    import server
    from api.routers import documents as documents_router

    incoming_pdf = tmp_path / "guide.pdf"
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 400)
    pdf.save(str(incoming_pdf))
    original_pdf = incoming_pdf.read_bytes()
    crop = tmp_path / "figure.png"
    crop.write_bytes(b"\x89PNG\r\n\x1a\nimmutable")
    library = tmp_path / "library"
    snapshot = ExpectedLibraryStore(library).publish({
        "MRSA Leitlinie": LibraryCandidate(
            canonical_markdown=(
                "# Guide\n\nSource text.\n\n"
                "![figure](/assets/MRSA Leitlinie/figure.png)"
            ),
            source=SourceCandidate(
                incoming_pdf,
                ocr_scale=2.0,
                assets=(SourceAssetCandidate(
                    "figure_1", crop, "image/png"
                ),),
            ),
        )
    })
    incoming_pdf.write_bytes(b"replacement")
    crop.write_bytes(b"replacement")
    monkeypatch.setattr(documents_router, "LIBRARY_DIR", library)
    client = TestClient(server.app)
    base = (
        f"/api/library/{snapshot.generation_id}/documents/"
        "MRSA%20Leitlinie"
    )
    listing = client.get("/api/documents").json()[0]
    assert listing["full_href"] == f"{base}/full"
    assert listing["source_href"] == f"{base}/pdf"
    assert listing["page_href"] == f"{base}/page/{{page}}"

    assert client.get(f"{base}/pdf").content == original_pdf
    rendered_markdown = client.get(f"{base}/full").json()["markdown"]
    assert "/assets/MRSA Leitlinie/figure.png" not in rendered_markdown
    assert f"{base}/assets/figure_1" in rendered_markdown
    assert client.get(f"{base}/assets/figure_1").content == (
        b"\x89PNG\r\n\x1a\nimmutable"
    )
    assert client.get(f"{base}/page/1").content.startswith(b"\x89PNG")


def test_page_geometry_uses_the_pin_scale_and_generation_source(tmp_path):
    pdfium = pytest.importorskip("pypdfium2")
    from api.pdf import rendered_page_size

    incoming_pdf = tmp_path / "guide.pdf"
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 400)
    pdf.save(str(incoming_pdf))
    snapshot = ExpectedLibraryStore(tmp_path / "library").publish({
        "guide": LibraryCandidate(
            canonical_markdown="# Guide\n\nSource text.",
            source=SourceCandidate(incoming_pdf, ocr_scale=2.0),
        )
    })
    source = snapshot.document("guide").source

    assert rendered_page_size(source, 1, pin_scale=2.0) == (400.0, 800.0)
    assert rendered_page_size(source, 1, pin_scale=4.0) is None


def test_publish_rejects_provenance_without_an_immutable_source(tmp_path):
    pin = ProvenancePin(
        version=2,
        document="guide",
        node_id="guide",
        spans=(SourceSpan(1, 0, 7, PixelBox(10, 20, 100, 80)),),
        scale=2.0,
    )

    with pytest.raises(ValueError, match="pin without an immutable source"):
        ExpectedLibraryStore(tmp_path / "library").publish({
            "guide": LibraryCandidate(
                canonical_markdown=f"# Guide\n\n{emit_pin(pin)}\nContent"
            )
        })


@pytest.mark.parametrize("document_id", [".", "..", "a/b", "a\\b", "a\0b", "a\nb"])
def test_publish_rejects_nonportable_document_identity(tmp_path, document_id):
    with pytest.raises(ValueError, match="invalid document identity"):
        ExpectedLibraryStore(tmp_path / "library").publish({
            document_id: LibraryCandidate("# Guide\n\nContent.")
        })


@pytest.mark.parametrize("asset_id", [".", "..", "a/b", "a\\b", "a\0b", "a\nb"])
def test_publish_rejects_nonportable_asset_identity(tmp_path, asset_id):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.7\nsource\n")
    asset = tmp_path / "asset.png"
    asset.write_bytes(b"asset")
    with pytest.raises(ValueError, match="invalid source asset identity"):
        ExpectedLibraryStore(tmp_path / "library").publish({
            "guide": LibraryCandidate(
                "# Guide\n\nContent.",
                SourceCandidate(
                    pdf,
                    2.0,
                    (SourceAssetCandidate(asset_id, asset, "image/png"),),
                ),
            )
        })


def test_generation_identity_normalizes_integer_and_float_ocr_scale(tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.7\nsource\n")
    candidate = lambda scale: LibraryCandidate(
        "# Guide\n\nContent.", SourceCandidate(pdf, scale)
    )

    integer = ExpectedLibraryStore(tmp_path / "one").publish(
        {"guide": candidate(2)}
    )
    floating = ExpectedLibraryStore(tmp_path / "two").publish(
        {"guide": candidate(2.0)}
    )

    assert integer.generation_id == floating.generation_id


def test_generation_identity_normalizes_source_asset_order(tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.7\nsource\n")
    first = tmp_path / "first.png"
    first.write_bytes(b"first")
    second = tmp_path / "second.png"
    second.write_bytes(b"second")
    assets = (
        SourceAssetCandidate("first", first, "image/png"),
        SourceAssetCandidate("second", second, "image/png"),
    )

    ordered = ExpectedLibraryStore(tmp_path / "ordered").publish({
        "guide": LibraryCandidate(
            "# Guide\n\nContent.", SourceCandidate(pdf, 2.0, assets)
        )
    })
    reversed_order = ExpectedLibraryStore(tmp_path / "reversed").publish({
        "guide": LibraryCandidate(
            "# Guide\n\nContent.", SourceCandidate(pdf, 2.0, assets[::-1])
        )
    })

    assert ordered.generation_id == reversed_order.generation_id
    assert [asset.asset_id for asset in ordered.document("guide").source.assets] == [
        "first",
        "second",
    ]


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan"), True])
def test_source_candidate_rejects_an_invalid_ocr_scale(tmp_path, scale):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"pdf")
    with pytest.raises(ValueError, match="ocr_scale"):
        SourceCandidate(pdf, scale)


def test_publish_rejects_a_pin_for_a_different_document(tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"pdf")
    pin = ProvenancePin(
        2,
        "other",
        "guide",
        (SourceSpan(1, 0, 7, PixelBox(1, 1, 2, 2)),),
        2.0,
    )
    with pytest.raises(ValueError, match="pin document"):
        ExpectedLibraryStore(tmp_path / "library").publish({
            "guide": LibraryCandidate(
                f"# Guide\n\n{emit_pin(pin)}\nContent",
                SourceCandidate(pdf, 2.0),
            )
        })


def test_publish_rejects_a_pin_for_a_different_node(tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"pdf")
    pin = ProvenancePin(
        2,
        "guide",
        "other-node",
        (SourceSpan(1, 0, 7, PixelBox(1, 1, 2, 2)),),
        2.0,
    )
    with pytest.raises(ValueError, match="pin node identity"):
        ExpectedLibraryStore(tmp_path / "library").publish({
            "guide": LibraryCandidate(
                f"# Guide\n\n{emit_pin(pin)}\nContent",
                SourceCandidate(pdf, 2.0),
            )
        })


def test_deserializer_rejects_a_malformed_real_serialization():
    serialized = serialize_document((PageNode(
        node_id="guide",
        title="Guide",
        heading_level=1,
        line_idx=0,
        summary="Content.",
        content="Content.",
    ),))
    malformed = json.loads(serialized)
    del malformed[0]["title"]

    with pytest.raises(ValueError, match="invalid serialized document tree"):
        deserialize_document(json.dumps(malformed))


def test_publish_rejects_source_evidence_changed_after_ingest(tmp_path):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"original")
    manifest = tmp_path / "sources.json"
    write_source_candidates(manifest, {"guide": SourceCandidate(pdf, 2.0)})
    pdf.write_bytes(b"replacement")

    with pytest.raises(ValueError, match="changed since ingest"):
        ExpectedLibraryStore(tmp_path / "library").publish({
            "guide": LibraryCandidate(
                "# Guide\n\nContent.",
                read_source_candidates(manifest)["guide"],
            )
        })
