"""The operator review lifecycle is revisioned, expiring, and atomic."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from hypothesis import given, strategies as st
from PIL import Image

from modules.ingest._betteringest.betteringest import Asset, IngestedDoc
from modules.ingest._betteringest.ocr import Block
from modules.ingest.review import (
    NormalizedBox,
    ReviewConflict,
    ReviewExpired,
    ReviewNotReady,
    RegionEdit,
    ReviewRegion,
    ReviewStore,
    SessionState,
    TableState,
)
from modules.ingest.table_recognition import TableRecognitionUnavailable
from pageindex.library import ExpectedLibraryStore, LibraryCandidate

pdfium = pytest.importorskip("pypdfium2")


def _pdf(path, pages=2):
    document = pdfium.PdfDocument.new()
    for _ in range(pages):
        document.new_page(200, 400)
    document.save(str(path))


def _doc(tmp_path, kind="figure") -> IngestedDoc:
    pdf = tmp_path / "source.pdf"
    _pdf(pdf)
    crop = tmp_path / f"{kind}_1.png"
    Image.new("RGB", (80, 120), "white").save(crop)
    caption = f"{kind.capitalize()} 1: Example"
    markdown = "\n".join([
        "# Guideline", "", "## Treatment", "", "Use verified guidance.", "",
        f"![{kind} 1](assets/{crop.name})", "",
    ])
    return IngestedDoc(
        doc_name="guide",
        title="Guideline",
        pdf_path=str(pdf),
        md_path="",
        markdown=markdown,
        assets=[Asset(
            asset_id=f"{kind}_1",
            type=kind,
            number=1,
            caption=caption,
            page=2,
            image=str(crop),
            bbox=[20, 40, 180, 280],
        )],
        tree=None,
        blocks=[
            Block("doc_title", "Guideline", 0, (10, 10, 150, 30)),
            Block("paragraph_title", "Treatment", 0, (10, 50, 150, 70)),
            Block("text", "Use verified guidance.", 0, (10, 80, 180, 120)),
        ],
        ladder_diag={},
        ocr_scale=2.0,
    )


@pytest.fixture
def store(tmp_path):
    return ReviewStore(tmp_path / "reviews", tmp_path / "library")


def test_figure_only_session_is_ready_and_persists(store, tmp_path):
    session = store.create([_doc(tmp_path)])
    assert session.state is SessionState.READY
    reopened = store.get(session.session_id)
    assert reopened == session
    assert (store.root / session.session_id / "guide" / "source.pdf").is_file()


def test_table_requires_recognition_or_explicit_failure_acknowledgement(store, tmp_path):
    session = store.create([_doc(tmp_path, "table")])
    assert session.state is SessionState.DRAFT
    with pytest.raises(ReviewNotReady):
        store.publish(session.session_id, session.revision)

    def unavailable(_path):
        raise TableRecognitionUnavailable("model unavailable")

    failed = store.recognize_table(
        session.session_id, "guide", "table_1", session.revision, unavailable
    )
    table = failed.documents[0].regions[0]
    assert table.table_state is TableState.FAILED
    assert failed.state is SessionState.DRAFT

    ready = store.acknowledge_table_failure(
        session.session_id, "guide", "table_1", failed.revision
    )
    assert ready.state is SessionState.READY


def test_successful_table_result_becomes_searchable_content(store, tmp_path):
    session = store.create([_doc(tmp_path, "table")])
    recognized = store.recognize_table(
        session.session_id,
        "guide",
        "table_1",
        session.revision,
        lambda _path: SimpleNamespace(
            markdown="| Agent | Duration |\n|---|---|\n| A | 5 min |",
            payload={"kind": "table"},
        ),
    )
    published, snapshot = store.publish(session.session_id, recognized.revision)
    assert published.state is SessionState.PUBLISHED
    assert "Duration" in snapshot.document("guide").canonical_markdown


def test_geometry_interface_preserves_only_unchanged_server_table_outcomes(store, tmp_path):
    session = store.create([_doc(tmp_path, "table")])
    recognized = store.recognize_table(
        session.session_id,
        "guide",
        "table_1",
        session.revision,
        lambda _path: SimpleNamespace(markdown="| verified |", payload={"ok": True}),
    )
    current = recognized.documents[0].regions[0]
    unchanged = RegionEdit.model_validate({
        **current.model_dump(include={
            "region_id", "kind", "page", "box", "caption",
            "asset_filename", "deleted",
        }),
        "caption": "Operator-confirmed table",
    })
    retained = store.replace_regions(
        session.session_id, "guide", recognized.revision, [unchanged]
    )
    assert retained.documents[0].regions[0].table_state is TableState.RECOGNIZED

    moved = unchanged.model_copy(update={
        "box": NormalizedBox(x=0.1, y=0.1, width=0.2, height=0.2),
    })
    reset = store.replace_regions(
        session.session_id, "guide", retained.revision, [moved]
    )
    assert reset.documents[0].regions[0].table_state is TableState.PENDING
    assert reset.state is SessionState.DRAFT


def test_stale_revision_cannot_overwrite_newer_geometry(store, tmp_path):
    session = store.create([_doc(tmp_path)])
    region = session.documents[0].regions[0]
    changed = region.model_copy(update={
        "box": NormalizedBox(x=0.1, y=0.1, width=0.4, height=0.4)
    })
    updated = store.replace_regions(
        session.session_id, "guide", session.revision, [changed]
    )
    assert updated.revision == session.revision + 1
    with pytest.raises(ReviewConflict, match="stale"):
        store.replace_regions(
            session.session_id, "guide", session.revision, [region]
        )


def test_rectangle_validation_is_strict_and_contained():
    with pytest.raises(ValueError):
        NormalizedBox.model_validate({
            "x": "0", "y": 0.0, "width": 0.5, "height": 0.5
        })
    with pytest.raises(ValueError, match="contained"):
        NormalizedBox(x=0.8, y=0.1, width=0.3, height=0.2)
    with pytest.raises(ValueError):
        ReviewRegion(
            region_id="bad/id",
            kind="figure",
            page=1,
            box=NormalizedBox(x=0.1, y=0.1, width=0.2, height=0.2),
            caption="Example",
            asset_filename="figure.png",
        )


@given(
    x=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    width=st.floats(min_value=1e-9, max_value=1, allow_nan=False, allow_infinity=False),
    height=st.floats(min_value=1e-9, max_value=1, allow_nan=False, allow_infinity=False),
)
def test_rectangle_acceptance_exactly_matches_page_containment(x, y, width, height):
    contained = x + width <= 1 and y + height <= 1
    try:
        box = NormalizedBox(x=x, y=y, width=width, height=height)
    except ValueError:
        assert not contained
    else:
        assert contained
        assert box.x + box.width <= 1
        assert box.y + box.height <= 1


def test_expired_session_is_fail_closed(tmp_path):
    now = [datetime(2026, 7, 17, tzinfo=UTC)]
    store = ReviewStore(
        tmp_path / "reviews", tmp_path / "library", clock=lambda: now[0]
    )
    session = store.create([_doc(tmp_path)])
    now[0] += timedelta(hours=25)
    with pytest.raises(ReviewExpired):
        store.get(session.session_id)


def test_publication_replaces_only_reviewed_documents(store, tmp_path):
    previous_pdf = tmp_path / "previous.pdf"
    _pdf(previous_pdf, pages=1)
    ExpectedLibraryStore(store.library.root).publish({
        "existing": LibraryCandidate("# Existing\n\nStable body."),
    })
    session = store.create([_doc(tmp_path)])
    _, snapshot = store.publish(session.session_id, session.revision)
    assert {doc.document_id for doc in snapshot.documents} == {"existing", "guide"}
    assert snapshot.document("existing").canonical_markdown.endswith("Stable body.")


def test_cancelled_session_cannot_be_published(store, tmp_path):
    session = store.create([_doc(tmp_path)])
    cancelled = store.cancel(session.session_id, session.revision)
    assert cancelled.state is SessionState.CANCELLED
    with pytest.raises(ReviewConflict):
        store.publish(session.session_id, cancelled.revision)
