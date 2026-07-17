"""Local operator interface for revisioned ingest review and publication."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from modules.ingest.review import (
    NormalizedBox,
    RegionEdit,
    ReviewError,
    ReviewSession,
    ReviewStore,
    SessionState,
    TableState,
)
from paths import LIBRARY_DIR, REVIEW_DIR

router = APIRouter()
review_store = ReviewStore(REVIEW_DIR, LIBRARY_DIR)


class ReviewRegionView(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    region_id: str
    kind: Literal["figure", "table"]
    page: int
    box: NormalizedBox
    caption: str
    asset_filename: str
    deleted: bool
    table_state: TableState | None
    table_error: str | None
    table_failure_acknowledged: bool


class ReviewDocumentView(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: str
    title: str
    page_count: int
    regions: list[ReviewRegionView]
    page_href_template: str


class ReviewSessionView(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: str
    revision: int
    state: SessionState
    expires_at: str
    documents: list[ReviewDocumentView]
    published_generation_id: str | None


class ReplaceRegionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expected_revision: int = Field(ge=1)
    regions: list[RegionEdit]


class RevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expected_revision: int = Field(ge=1)


class ActiveReviewView(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    review_id: str | None


def session_view(session: ReviewSession) -> ReviewSessionView:
    return ReviewSessionView(
        session_id=session.session_id,
        revision=session.revision,
        state=session.state,
        expires_at=session.expires_at.isoformat(),
        documents=[ReviewDocumentView(
            document_id=document.document_id,
            title=document.title,
            page_count=document.page_count,
            regions=[ReviewRegionView(
                region_id=region.region_id,
                kind=region.kind,
                page=region.page,
                box=region.box,
                caption=region.caption,
                asset_filename=region.asset_filename,
                deleted=region.deleted,
                table_state=region.table_state,
                table_error=(
                    "Local table recognition did not produce structure."
                    if region.table_error else None
                ),
                table_failure_acknowledged=region.table_failure_acknowledged,
            ) for region in document.regions],
            page_href_template=(
                f"/api/ingest/reviews/{session.session_id}/documents/"
                f"{document.document_id}/pages/{{page}}.png"
            ),
        ) for document in session.documents],
        published_generation_id=session.published_generation_id,
    )


def _error(exc: ReviewError) -> JSONResponse:
    return JSONResponse(
        {"code": exc.code, "message": str(exc)}, status_code=exc.status_code
    )


@router.get("/api/ingest/review-active", response_model=ActiveReviewView)
def get_active_review():
    try:
        session = review_store.active()
        return ActiveReviewView(
            review_id=session.session_id if session is not None else None
        )
    except ReviewError as exc:
        return _error(exc)


@router.get(
    "/api/ingest/reviews/{session_id}", response_model=ReviewSessionView
)
def get_review(session_id: str):
    try:
        return session_view(review_store.get(session_id, allow_terminal=True))
    except ReviewError as exc:
        return _error(exc)


@router.get(
    "/api/ingest/reviews/{session_id}/documents/{document_id}/pages/{page}.png"
)
def get_review_page(session_id: str, document_id: str, page: int):
    try:
        return Response(
            review_store.page_png(session_id, document_id, page),
            media_type="image/png",
        )
    except ReviewError as exc:
        return _error(exc)


@router.put(
    "/api/ingest/reviews/{session_id}/documents/{document_id}/regions",
    response_model=ReviewSessionView,
)
def replace_regions(
    session_id: str, document_id: str, request: ReplaceRegionsRequest
):
    try:
        return session_view(review_store.replace_regions(
            session_id,
            document_id,
            request.expected_revision,
            request.regions,
        ))
    except ReviewError as exc:
        return _error(exc)


@router.post(
    "/api/ingest/reviews/{session_id}/documents/{document_id}/regions/"
    "{region_id}/recognize-table",
    response_model=ReviewSessionView,
)
def recognize_table(
    session_id: str,
    document_id: str,
    region_id: str,
    request: RevisionRequest,
):
    try:
        return session_view(review_store.recognize_table(
            session_id,
            document_id,
            region_id,
            request.expected_revision,
        ))
    except ReviewError as exc:
        return _error(exc)


@router.post(
    "/api/ingest/reviews/{session_id}/documents/{document_id}/regions/"
    "{region_id}/acknowledge-table-failure",
    response_model=ReviewSessionView,
)
def acknowledge_table_failure(
    session_id: str,
    document_id: str,
    region_id: str,
    request: RevisionRequest,
):
    try:
        return session_view(review_store.acknowledge_table_failure(
            session_id,
            document_id,
            region_id,
            request.expected_revision,
        ))
    except ReviewError as exc:
        return _error(exc)


@router.post(
    "/api/ingest/reviews/{session_id}/confirm", response_model=ReviewSessionView
)
def confirm_review(session_id: str, request: RevisionRequest):
    try:
        session, _snapshot = review_store.publish(
            session_id, request.expected_revision
        )
        return session_view(session)
    except ReviewError as exc:
        return _error(exc)


@router.post(
    "/api/ingest/reviews/{session_id}/cancel", response_model=ReviewSessionView
)
def cancel_review(session_id: str, request: RevisionRequest):
    try:
        return session_view(review_store.cancel(
            session_id, request.expected_revision
        ))
    except ReviewError as exc:
        return _error(exc)
