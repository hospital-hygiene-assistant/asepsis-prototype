"""Persistent operator review of PDF assets before immutable publication.

The interface is the complete review lifecycle: candidate creation, revisioned
geometry, table outcomes, expiry, and atomic Expected-library publication.
"""

from __future__ import annotations

import math
import mimetypes
import os
import re
import secrets
import shutil
import threading
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modules.ingest._betteringest.betteringest import Asset, IngestedDoc
from modules.ingest._betteringest.ocr import Block
from modules.ingest._massage import massage
from pageindex.library import (
    ExpectedLibraryNotBuilt,
    ExpectedLibrarySnapshot,
    ExpectedLibraryStore,
    LibraryCandidate,
    SourceAssetCandidate,
    SourceCandidate,
    validate_library_segment,
)

SESSION_VERSION = 1
SESSION_LIFETIME = timedelta(hours=24)
_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
_REGION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_ASSET_LINK = re.compile(r"^!\[(figure|table) (\d+)\]\(assets/([^)]+)\)\s*$")


def _has_non_asset_body(markdown: str) -> bool:
    return any(
        line.strip()
        and not line.lstrip().startswith("#")
        and _ASSET_LINK.fullmatch(line) is None
        for line in markdown.splitlines()
    )


def _valid_region_id(value: str) -> str:
    if not _REGION_ID.fullmatch(value):
        raise ValueError("invalid region identity")
    return value


def _valid_asset_filename(value: str) -> str:
    validate_library_segment(value, "asset filename")
    if Path(value).suffix.lower() != ".png":
        raise ValueError("review assets must be PNG files")
    return value


class ReviewError(RuntimeError):
    status_code = 400
    code = "invalid_review"


class ReviewNotFound(ReviewError):
    status_code = 404
    code = "review_not_found"


class ReviewExpired(ReviewError):
    status_code = 410
    code = "review_expired"


class ReviewConflict(ReviewError):
    status_code = 409
    code = "review_conflict"


class ReviewNotReady(ReviewError):
    status_code = 409
    code = "review_not_ready"


class SessionState(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    PUBLISHED = "published"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TableState(str, Enum):
    PENDING = "pending"
    RECOGNIZED = "recognized"
    FAILED = "failed"


class NormalizedBox(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @field_validator("x", "y", "width", "height")
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("rectangle values must be finite")
        return value

    @model_validator(mode="after")
    def contained(self):
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("rectangle must be contained within the page")
        return self


class ReviewRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    region_id: str
    kind: Literal["figure", "table"]
    page: int = Field(ge=1)
    box: NormalizedBox
    caption: str = Field(min_length=1, max_length=500)
    asset_filename: str
    deleted: bool = False
    table_state: TableState | None = None
    table_markdown: str | None = None
    table_payload: dict | None = None
    table_error: str | None = None
    table_failure_acknowledged: bool = False

    @field_validator("region_id")
    @classmethod
    def valid_region_id(cls, value: str) -> str:
        return _valid_region_id(value)

    @field_validator("asset_filename")
    @classmethod
    def valid_filename(cls, value: str) -> str:
        return _valid_asset_filename(value)

    @model_validator(mode="after")
    def table_fields_match_kind(self):
        if self.kind == "figure":
            if any((self.table_state, self.table_markdown, self.table_payload,
                    self.table_error, self.table_failure_acknowledged)):
                raise ValueError("figure regions cannot carry table outcomes")
        elif self.table_state is None:
            self.table_state = TableState.PENDING
        if self.table_state is TableState.RECOGNIZED and not self.table_markdown:
            raise ValueError("recognized tables require structured output")
        if self.table_failure_acknowledged and self.table_state is not TableState.FAILED:
            raise ValueError("only failed table recognition can be acknowledged")
        return self


class RegionEdit(BaseModel):
    """The complete client-editable region shape; outcomes stay server-owned."""

    model_config = ConfigDict(extra="forbid", strict=True)

    region_id: str
    kind: Literal["figure", "table"]
    page: int = Field(ge=1)
    box: NormalizedBox
    caption: str = Field(min_length=1, max_length=500)
    asset_filename: str
    deleted: bool = False

    @field_validator("region_id")
    @classmethod
    def valid_region_id(cls, value: str) -> str:
        return _valid_region_id(value)

    @field_validator("asset_filename")
    @classmethod
    def valid_filename(cls, value: str) -> str:
        return _valid_asset_filename(value)


class ReviewBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    label: str
    text: str
    page: int = Field(ge=0)
    bbox: tuple[float, float, float, float]


class ReviewDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: str
    title: str
    pdf_filename: str
    page_count: int = Field(ge=1)
    asset_revision: int = Field(default=1, ge=1)
    ocr_scale: float = Field(gt=0)
    markdown: str
    blocks: list[ReviewBlock]
    regions: list[ReviewRegion]

    @field_validator("document_id")
    @classmethod
    def valid_document_id(cls, value: str) -> str:
        return validate_library_segment(value, "document identity")

    @field_validator("pdf_filename")
    @classmethod
    def valid_pdf_filename(cls, value: str) -> str:
        validate_library_segment(value, "PDF filename")
        if Path(value).suffix.lower() != ".pdf":
            raise ValueError("review source must be a PDF")
        return value

    @model_validator(mode="after")
    def unique_regions(self):
        ids = [region.region_id for region in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("region identities must be unique within a document")
        return self


class ReviewSession(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = SESSION_VERSION
    session_id: str
    revision: int = Field(ge=1)
    state: SessionState
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    documents: list[ReviewDocument]
    published_generation_id: str | None = None

    @field_validator("session_id")
    @classmethod
    def valid_session_id(cls, value: str) -> str:
        if not _SESSION_ID.fullmatch(value):
            raise ValueError("invalid review-session identity")
        return value

    @model_validator(mode="after")
    def unique_documents(self):
        ids = [document.document_id for document in self.documents]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("review documents must be non-empty and unique")
        return self


class ReviewStore:
    """One process-safe, atomically persisted review lifecycle."""

    def __init__(
        self,
        root: Path,
        library_root: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = root
        self.library = ExpectedLibraryStore(library_root)
        self.clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()

    def _session_dir(self, session_id: str) -> Path:
        if not _SESSION_ID.fullmatch(session_id):
            raise ReviewNotFound("unknown review session")
        return self.root / session_id

    def _state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.json"

    def _assets_dir(
        self, session_id: str, document: ReviewDocument
    ) -> Path:
        return (
            self._session_dir(session_id)
            / document.document_id
            / f"assets-r{document.asset_revision}"
        )

    def _write(self, session: ReviewSession) -> None:
        directory = self._session_dir(session.session_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = directory / f".session-{secrets.token_hex(8)}.json"
        temporary.write_text(
            session.model_dump_json(indent=2), encoding="utf-8"
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, directory / "session.json")

    def _read(self, session_id: str, *, allow_terminal: bool = False) -> ReviewSession:
        path = self._state_path(session_id)
        try:
            session = ReviewSession.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ReviewNotFound("unknown review session") from exc
        except Exception as exc:
            raise ReviewNotFound("review session is unreadable") from exc
        now = self.clock()
        if session.state not in (SessionState.PUBLISHED, SessionState.CANCELLED) and now >= session.expires_at:
            session.state = SessionState.EXPIRED
            session.updated_at = now
            session.revision += 1
            self._write(session)
        if session.state is SessionState.EXPIRED and not allow_terminal:
            raise ReviewExpired("review session has expired")
        if session.state in (SessionState.PUBLISHED, SessionState.CANCELLED) and not allow_terminal:
            raise ReviewConflict(f"review session is already {session.state.value}")
        return session

    @staticmethod
    def _page_dimensions(pdf_path: Path, page: int, scale: float) -> tuple[float, float]:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf_path))
        if page < 1 or page > len(document):
            raise ReviewError(f"page {page} is outside the source PDF")
        width, height = document[page - 1].get_size()
        return width * scale, height * scale

    @staticmethod
    def _embedded_region_text(
        pdf_path: Path, page: int, box: NormalizedBox
    ) -> str:
        """Return authoritative embedded PDF text inside a reviewed region."""
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf_path))
        if page < 1 or page > len(document):
            raise ReviewError(f"page {page} is outside the source PDF")
        pdf_page = document[page - 1]
        width, height = pdf_page.get_size()
        left = box.x * width
        right = (box.x + box.width) * width
        top = (1 - box.y) * height
        bottom = (1 - box.y - box.height) * height
        text = pdf_page.get_textpage().get_text_bounded(
            left, bottom, right, top
        )
        # PDFium uses U+0002 for a visible discretionary hyphen in some UKR
        # documents. Preserve that meaning while normalizing line endings.
        return ReviewStore._normalize_embedded_text(text)

    @staticmethod
    def _normalize_embedded_text(text: str) -> str:
        return (
            text.replace("\x02", "-")
            .replace("\ufffe", "-")
            .replace("\r\n", "\n")
            .strip()
        )

    @staticmethod
    def _embedded_document_guidance(
        pdf_path: Path, scale: float
    ) -> tuple[str, tuple[Block, ...]]:
        """Recover exact text when layout found only visual regions."""
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        texts: list[str] = []
        blocks: list[Block] = [
            Block("paragraph_title", "Source text", 0, (0, 0, 1, 1))
        ]
        for page_index, page in enumerate(pdf):
            text = ReviewStore._normalize_embedded_text(
                page.get_textpage().get_text_range()
            )
            if not text:
                continue
            width, height = page.get_size()
            texts.append(text)
            blocks.append(Block(
                "text",
                text,
                page_index,
                (0, 0, width * scale, height * scale),
            ))
        return "\n\n".join(texts), tuple(blocks)

    @staticmethod
    def _page_count(pdf_path: Path) -> int:
        import pypdfium2 as pdfium

        return len(pdfium.PdfDocument(str(pdf_path)))

    def create(self, docs: Iterable[IngestedDoc]) -> ReviewSession:
        with self._lock:
            prepared = list(docs)
            if not prepared:
                raise ReviewError("cannot review an empty ingest batch")
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            if self.active() is not None:
                raise ReviewConflict("an ingest review is already active")

            session_id = secrets.token_hex(16)
            directory = self._session_dir(session_id)
            directory.mkdir(mode=0o700)
            documents: list[ReviewDocument] = []
            try:
                for doc in prepared:
                    document_id = validate_library_segment(doc.doc_name, "document identity")
                    document_dir = directory / document_id
                    assets_dir = document_dir / "assets-r1"
                    assets_dir.mkdir(parents=True, mode=0o700)
                    source = Path(doc.pdf_path)
                    pdf_filename = "source.pdf"
                    shutil.copy2(source, document_dir / pdf_filename)
                    regions = []
                    for asset in doc.assets:
                        filename = Path(asset.image).name
                        validate_library_segment(filename, "asset filename")
                        shutil.copy2(asset.image, assets_dir / filename)
                        width, height = self._page_dimensions(
                            document_dir / pdf_filename, asset.page, doc.ocr_scale
                        )
                        x0, y0, x1, y1 = (float(value) for value in asset.bbox)
                        regions.append(ReviewRegion(
                            region_id=asset.asset_id,
                            kind=asset.type,
                            page=asset.page,
                            box=NormalizedBox(
                                x=x0 / width,
                                y=y0 / height,
                                width=(x1 - x0) / width,
                                height=(y1 - y0) / height,
                            ),
                            caption=asset.caption or f"{asset.type.capitalize()} (unlabeled)",
                            asset_filename=filename,
                            table_state=(TableState.PENDING if asset.type == "table" else None),
                        ))
                    documents.append(ReviewDocument(
                        document_id=document_id,
                        title=doc.title or document_id,
                        pdf_filename=pdf_filename,
                        page_count=self._page_count(document_dir / pdf_filename),
                        ocr_scale=float(doc.ocr_scale),
                        markdown=doc.markdown,
                        blocks=[ReviewBlock(
                            label=block.label,
                            text=block.text,
                            page=block.page,
                            bbox=tuple(float(value) for value in block.bbox),
                        ) for block in doc.blocks],
                        regions=regions,
                    ))
                now = self.clock()
                session = ReviewSession(
                    session_id=session_id,
                    revision=1,
                    state=SessionState.DRAFT,
                    created_at=now,
                    updated_at=now,
                    expires_at=now + SESSION_LIFETIME,
                    documents=documents,
                )
                self._refresh_state(session)
                self._write(session)
                return session
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                raise

    @staticmethod
    def _refresh_state(session: ReviewSession) -> None:
        tables = [
            region for document in session.documents for region in document.regions
            if not region.deleted and region.kind == "table"
        ]
        ready = all(
            region.table_state is TableState.RECOGNIZED
            or (
                region.table_state is TableState.FAILED
                and region.table_failure_acknowledged
            )
            for region in tables
        )
        session.state = SessionState.READY if ready else SessionState.DRAFT

    def get(self, session_id: str, *, allow_terminal: bool = False) -> ReviewSession:
        with self._lock:
            return self._read(session_id, allow_terminal=allow_terminal)

    def active(self) -> ReviewSession | None:
        """Return the single resumable review, if one exists."""
        with self._lock:
            if not self.root.is_dir():
                return None
            for child in sorted(self.root.iterdir()):
                if not child.is_dir() or not _SESSION_ID.fullmatch(child.name):
                    continue
                session = self._read(child.name, allow_terminal=True)
                if session.state not in (
                    SessionState.PUBLISHED,
                    SessionState.CANCELLED,
                    SessionState.EXPIRED,
                ):
                    return session
            return None

    def _assert_revision(self, session: ReviewSession, expected: int) -> None:
        if session.revision != expected:
            raise ReviewConflict(
                f"stale review revision {expected}; current revision is {session.revision}"
            )

    def _document(self, session: ReviewSession, document_id: str) -> ReviewDocument:
        try:
            return next(doc for doc in session.documents if doc.document_id == document_id)
        except StopIteration as exc:
            raise ReviewNotFound("unknown review document") from exc

    def _region(self, document: ReviewDocument, region_id: str) -> ReviewRegion:
        try:
            return next(region for region in document.regions if region.region_id == region_id)
        except StopIteration as exc:
            raise ReviewNotFound("unknown review region") from exc

    def page_png(self, session_id: str, document_id: str, page: int) -> bytes:
        with self._lock:
            session = self._read(session_id)
            document = self._document(session, document_id)
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(
                self._session_dir(session_id) / document_id / document.pdf_filename
            ))
            if page < 1 or page > len(pdf):
                raise ReviewNotFound("unknown review page")
            image = pdf[page - 1].render(scale=document.ocr_scale).to_pil()
            from io import BytesIO

            output = BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()

    def replace_regions(
        self,
        session_id: str,
        document_id: str,
        expected_revision: int,
        regions: list[RegionEdit],
    ) -> ReviewSession:
        with self._lock:
            session = self._read(session_id)
            self._assert_revision(session, expected_revision)
            document = self._document(session, document_id)
            ids = [region.region_id for region in regions]
            if len(ids) != len(set(ids)):
                raise ReviewError("region identities must be unique")
            existing = {region.region_id: region for region in document.regions}
            updated_regions = []
            for edit in regions:
                previous = existing.get(edit.region_id)
                evidence_changed = previous is None or any((
                    previous.kind != edit.kind,
                    previous.page != edit.page,
                    previous.box != edit.box,
                    previous.asset_filename != edit.asset_filename,
                ))
                retained = {}
                if previous is not None and not evidence_changed and edit.kind == "table":
                    retained = {
                        "table_state": previous.table_state,
                        "table_markdown": previous.table_markdown,
                        "table_payload": previous.table_payload,
                        "table_error": previous.table_error,
                        "table_failure_acknowledged": (
                            previous.table_failure_acknowledged
                        ),
                    }
                region = ReviewRegion(**edit.model_dump(), **retained)
                updated_regions.append(region)

            next_revision = session.revision + 1
            document_dir = self._session_dir(session_id) / document.document_id
            staged = document_dir / f".assets-r{next_revision}-{secrets.token_hex(8)}"
            final = document_dir / f"assets-r{next_revision}"
            # A process crash may leave this unreferenced directory after its
            # atomic rename but before session.json moved to the new revision.
            shutil.rmtree(final, ignore_errors=True)
            staged.mkdir(mode=0o700)
            try:
                for region in updated_regions:
                    self._write_crop(session, document, region, staged)
                os.replace(staged, final)
                document.regions = updated_regions
                document.asset_revision = next_revision
                session.revision = next_revision
                session.updated_at = self.clock()
                self._refresh_state(session)
                self._write(session)
                return session
            except Exception:
                shutil.rmtree(staged, ignore_errors=True)
                shutil.rmtree(final, ignore_errors=True)
                raise

    def _write_crop(
        self,
        session: ReviewSession,
        document: ReviewDocument,
        region: ReviewRegion,
        destination_dir: Path,
    ) -> None:
        if region.deleted:
            return
        import pypdfium2 as pdfium

        pdf_path = (
            self._session_dir(session.session_id)
            / document.document_id
            / document.pdf_filename
        )
        pdf = pdfium.PdfDocument(str(pdf_path))
        if region.page > len(pdf):
            raise ReviewError(f"page {region.page} is outside the source PDF")
        image = pdf[region.page - 1].render(scale=document.ocr_scale).to_pil()
        left = round(region.box.x * image.width)
        top = round(region.box.y * image.height)
        right = round((region.box.x + region.box.width) * image.width)
        bottom = round((region.box.y + region.box.height) * image.height)
        if right <= left or bottom <= top:
            raise ReviewError("review rectangle produces an empty crop")
        destination = destination_dir / region.asset_filename
        temporary = destination.with_name(f".{secrets.token_hex(8)}.png")
        image.crop((left, top, right, bottom)).save(temporary, format="PNG")
        os.replace(temporary, destination)

    def recognize_table(
        self,
        session_id: str,
        document_id: str,
        region_id: str,
        expected_revision: int,
        recognizer=None,
    ) -> ReviewSession:
        from modules.ingest.table_recognition import (
            TableRecognitionUnavailable,
            recognize_table,
        )

        with self._lock:
            session = self._read(session_id)
            self._assert_revision(session, expected_revision)
            document = self._document(session, document_id)
            region = self._region(document, region_id)
            if region.kind != "table" or region.deleted:
                raise ReviewError("only active table regions can be recognized")
            path = (
                self._assets_dir(session_id, document) / region.asset_filename
            )
            try:
                result = (recognizer or recognize_table)(path)
                region.table_state = TableState.RECOGNIZED
                region.table_markdown = result.markdown
                region.table_payload = result.payload
                region.table_error = None
                region.table_failure_acknowledged = False
            except TableRecognitionUnavailable as exc:
                region.table_state = TableState.FAILED
                region.table_markdown = None
                region.table_payload = None
                region.table_error = str(exc)
                region.table_failure_acknowledged = False
            session.revision += 1
            session.updated_at = self.clock()
            self._refresh_state(session)
            self._write(session)
            return session

    def acknowledge_table_failure(
        self,
        session_id: str,
        document_id: str,
        region_id: str,
        expected_revision: int,
    ) -> ReviewSession:
        with self._lock:
            session = self._read(session_id)
            self._assert_revision(session, expected_revision)
            region = self._region(self._document(session, document_id), region_id)
            if region.table_state is not TableState.FAILED:
                raise ReviewError("only a failed table result can be acknowledged")
            region.table_failure_acknowledged = True
            session.revision += 1
            session.updated_at = self.clock()
            self._refresh_state(session)
            self._write(session)
            return session

    def cancel(self, session_id: str, expected_revision: int) -> ReviewSession:
        with self._lock:
            session = self._read(session_id)
            self._assert_revision(session, expected_revision)
            session.state = SessionState.CANCELLED
            session.revision += 1
            session.updated_at = self.clock()
            self._write(session)
            return session

    def _review_candidate(
        self, session: ReviewSession, document: ReviewDocument
    ) -> LibraryCandidate:
        directory = self._session_dir(session.session_id) / document.document_id
        assets_dir = self._assets_dir(session.session_id, document)
        pdf_path = directory / document.pdf_filename
        active = [region for region in document.regions if not region.deleted]
        active_names = {region.asset_filename for region in active}
        markdown_lines = [
            line for line in document.markdown.splitlines()
            if not (
                (match := _ASSET_LINK.fullmatch(line))
                and match.group(3) not in active_names
            )
        ]
        linked = {
            match.group(3)
            for line in markdown_lines
            if (match := _ASSET_LINK.fullmatch(line))
        }
        for region in active:
            if region.asset_filename not in linked:
                number = next(
                    (int(part) for part in region.region_id.split("_") if part.isdigit()),
                    len(linked) + 1,
                )
                markdown_lines.append(
                    f"![{region.kind} {number}](assets/{region.asset_filename})"
                )
                linked.add(region.asset_filename)

        blocks = [Block(
            label=block.label,
            text=block.text,
            page=block.page,
            bbox=block.bbox,
        ) for block in document.blocks]
        if not _has_non_asset_body("\n".join(markdown_lines)):
            source_text, source_blocks = self._embedded_document_guidance(
                pdf_path, document.ocr_scale
            )
            if source_text:
                markdown_lines.extend(["", "## Source text", "", source_text])
                blocks.extend(source_blocks)

        assets = []
        for index, region in enumerate(active, start=1):
            width, height = self._page_dimensions(
                pdf_path, region.page, document.ocr_scale
            )
            description = region.table_markdown or ""
            if region.kind == "table":
                description = (
                    self._embedded_region_text(pdf_path, region.page, region.box)
                    or description
                )
            assets.append(Asset(
                asset_id=region.region_id,
                type=region.kind,
                number=index,
                caption=region.caption,
                page=region.page,
                image=str(assets_dir / region.asset_filename),
                description=description,
                bbox=[
                    region.box.x * width,
                    region.box.y * height,
                    (region.box.x + region.box.width) * width,
                    (region.box.y + region.box.height) * height,
                ],
            ))
        ingested = IngestedDoc(
            doc_name=document.document_id,
            title=document.title,
            pdf_path=str(directory / document.pdf_filename),
            md_path="",
            markdown="\n".join(markdown_lines),
            assets=assets,
            tree=None,
            blocks=blocks,
            ladder_diag={},
            ocr_scale=document.ocr_scale,
        )
        canonical = massage(
            ingested, document.document_id,
            f"/assets/{document.document_id}",
        )
        return LibraryCandidate(
            canonical_markdown=canonical,
            source=SourceCandidate(
                pdf_path=directory / document.pdf_filename,
                ocr_scale=document.ocr_scale,
                assets=tuple(SourceAssetCandidate(
                    asset_id=region.region_id,
                    path=assets_dir / region.asset_filename,
                    media_type=(
                        mimetypes.guess_type(region.asset_filename)[0]
                        or "image/png"
                    ),
                ) for region in active),
            ),
        )

    def publish(
        self, session_id: str, expected_revision: int
    ) -> tuple[ReviewSession, ExpectedLibrarySnapshot]:
        with self._lock:
            session = self._read(session_id)
            self._assert_revision(session, expected_revision)
            self._refresh_state(session)
            if session.state is not SessionState.READY:
                raise ReviewNotReady(
                    "every active table must be recognized or have its failure acknowledged"
                )
            try:
                candidates = self.library.open_current().publication_candidates()
            except ExpectedLibraryNotBuilt:
                candidates = {}
            for document in session.documents:
                candidates[document.document_id] = self._review_candidate(session, document)
            snapshot = self.library.publish(candidates)
            session.state = SessionState.PUBLISHED
            session.published_generation_id = snapshot.generation_id
            session.revision += 1
            session.updated_at = self.clock()
            self._write(session)
            return session, snapshot

    def cleanup(self) -> int:
        """Expire active sessions and remove terminal directories after 24h."""
        with self._lock:
            if not self.root.is_dir():
                return 0
            removed = 0
            now = self.clock()
            for child in self.root.iterdir():
                if not child.is_dir() or not _SESSION_ID.fullmatch(child.name):
                    continue
                try:
                    session = self._read(child.name, allow_terminal=True)
                except ReviewError:
                    continue
                if now >= session.expires_at:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            return removed
