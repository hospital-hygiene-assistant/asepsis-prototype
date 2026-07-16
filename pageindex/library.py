"""Immutable Expected library generations and their source evidence."""

import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import quote

from .nodes import PageNode, parse_document
from .serialization import deserialize_document, serialize_document


class ExpectedLibraryNotBuilt(RuntimeError):
    """No complete Expected library generation has been published."""


class ExpectedLibraryCorrupt(RuntimeError):
    """A published generation no longer matches its immutable manifest."""


def validate_library_segment(value: str, label: str = "identity") -> str:
    """Validate one portable decoded document or asset path segment."""
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _normalize_ocr_scale(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("ocr_scale must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("ocr_scale must be a finite positive number")
    return result


def _generation_identity(document_records: list[dict]) -> str:
    canonical = json.dumps(
        {"documents": document_records},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class SourceCandidate:
    pdf_path: Path
    ocr_scale: float
    assets: tuple["SourceAssetCandidate", ...] = ()
    _expected_sha256: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ocr_scale", _normalize_ocr_scale(self.ocr_scale))


@dataclass(frozen=True)
class SourceAssetCandidate:
    asset_id: str
    path: Path
    media_type: str
    _expected_sha256: str | None = field(default=None, init=False, repr=False)


@dataclass(frozen=True)
class LibraryCandidate:
    canonical_markdown: str
    source: SourceCandidate | None = None


@dataclass(frozen=True)
class SourceDocument:
    sha256: str
    pdf_path: Path
    ocr_scale: float
    href: str
    assets: tuple["SourceAsset", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ocr_scale", _normalize_ocr_scale(self.ocr_scale))


@dataclass(frozen=True)
class SourceAsset:
    asset_id: str
    filename: str
    sha256: str
    path: Path
    media_type: str
    href: str


@dataclass(frozen=True)
class LibraryDocument:
    document_id: str
    index_path: Path
    canonical_markdown: str
    source: SourceDocument | None


@dataclass(frozen=True)
class ExpectedLibrarySnapshot:
    generation_id: str
    documents: tuple[LibraryDocument, ...]

    def document(self, document_id: str) -> LibraryDocument:
        try:
            return next(
                document
                for document in self.documents
                if document.document_id == document_id
            )
        except StopIteration as exc:
            raise KeyError(document_id) from exc


class ExpectedLibraryStore:
    """Publish and open complete immutable Expected library generations."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.generations = root / "generations"
        self.objects = root / "objects"
        self.current = root / "current.json"

    def publish(
        self, candidates: Mapping[str, LibraryCandidate]
    ) -> ExpectedLibrarySnapshot:
        if not candidates:
            raise ValueError("cannot publish an empty Expected library")
        ordered = tuple(sorted(candidates.items()))
        source_payloads: dict[str, bytes] = {}
        asset_payloads: dict[tuple[str, str], bytes] = {}
        index_payloads: dict[str, str] = {}
        for document_id, candidate in ordered:
            validate_library_segment(document_id, "document identity")
            nodes = tuple(parse_document(candidate.canonical_markdown, document_id))
            index_json = serialize_document(nodes)
            deserialize_document(index_json)
            index_payloads[document_id] = index_json
            if candidate.source is None and any(
                node.pin is not None for node in _walk_nodes(nodes)
            ):
                raise ValueError(
                    f"document {document_id!r} has a provenance pin without "
                    "an immutable source"
                )
            if candidate.source is not None:
                declared_assets = {
                    asset.asset_id for asset in candidate.source.assets
                }
                for node in _walk_nodes(nodes):
                    if (
                        node.pin is not None
                        and node.pin.scale != candidate.source.ocr_scale
                    ):
                        raise ValueError(
                            f"pin scale for {document_id!r} does not match "
                            "its immutable source"
                        )
                    if (
                        node.pin is not None
                        and node.pin.asset is not None
                        and node.pin.asset.asset_id not in declared_assets
                    ):
                        raise ValueError(
                            f"pin for {document_id!r} references undeclared "
                            f"source asset {node.pin.asset.asset_id!r}"
                        )
            if candidate.source is not None:
                data = candidate.source.pdf_path.read_bytes()
                actual = hashlib.sha256(data).hexdigest()
                if (
                    candidate.source._expected_sha256 is not None
                    and actual != candidate.source._expected_sha256
                ):
                    raise ValueError(
                        f"source PDF for {document_id!r} changed since ingest"
                    )
                source_payloads[document_id] = data
                seen_assets: set[str] = set()
                for asset in candidate.source.assets:
                    try:
                        validate_library_segment(
                            asset.asset_id, "source asset identity"
                        )
                    except ValueError:
                        raise ValueError(
                            f"invalid source asset identity: {asset.asset_id!r}"
                        )
                    if asset.asset_id in seen_assets:
                        raise ValueError(
                            f"invalid source asset identity: {asset.asset_id!r}"
                        )
                    seen_assets.add(asset.asset_id)
                    asset_data = asset.path.read_bytes()
                    asset_digest = hashlib.sha256(asset_data).hexdigest()
                    if (
                        asset._expected_sha256 is not None
                        and asset_digest != asset._expected_sha256
                    ):
                        raise ValueError(
                            f"source asset {asset.asset_id!r} for "
                            f"{document_id!r} changed since ingest"
                        )
                    asset_payloads[(document_id, asset.asset_id)] = asset_data

        self.objects.mkdir(parents=True, exist_ok=True)
        source_records: dict[str, dict] = {}
        for document_id, data in source_payloads.items():
            source = candidates[document_id].source
            assert source is not None
            source_sha256 = hashlib.sha256(data).hexdigest()
            pdf_dir = self.objects / "pdf"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            destination = pdf_dir / f"{source_sha256}.pdf"
            if not destination.exists():
                temporary = pdf_dir / f".{uuid.uuid4().hex}.pdf"
                temporary.write_bytes(data)
                os.replace(temporary, destination)
            source_records[document_id] = {
                "sha256": source_sha256,
                "ocr_scale": source.ocr_scale,
                "assets": [{
                    "asset_id": asset.asset_id,
                    "filename": asset.path.name,
                    "sha256": hashlib.sha256(
                        asset_payloads[(document_id, asset.asset_id)]
                    ).hexdigest(),
                    "media_type": asset.media_type,
                } for asset in sorted(
                    source.assets, key=lambda item: item.asset_id
                )],
            }

        assets_dir = self.objects / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for data in asset_payloads.values():
            asset_sha256 = hashlib.sha256(data).hexdigest()
            destination = assets_dir / asset_sha256
            if not destination.exists():
                temporary = assets_dir / f".{uuid.uuid4().hex}"
                temporary.write_bytes(data)
                os.replace(temporary, destination)

        document_records = [
            {
                "document_id": document_id,
                "index_sha256": hashlib.sha256(
                    index_payloads[document_id].encode("utf-8")
                ).hexdigest(),
                "markdown_sha256": hashlib.sha256(
                    candidate.canonical_markdown.encode("utf-8")
                ).hexdigest(),
                "source": source_records.get(document_id),
            }
            for document_id, candidate in ordered
        ]
        generation_id = _generation_identity(document_records)

        self.generations.mkdir(parents=True, exist_ok=True)
        staging = self.generations / f".building-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            for document_id, candidate in ordered:
                (staging / f"{document_id}.json").write_text(
                    index_payloads[document_id], encoding="utf-8"
                )
                (staging / f"{document_id}.md").write_text(
                    candidate.canonical_markdown, encoding="utf-8"
                )
            (staging / "manifest.json").write_text(
                json.dumps({
                    "generation_id": generation_id,
                    "documents": document_records,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            destination = self.generations / generation_id
            if destination.exists():
                shutil.rmtree(staging)
            else:
                os.replace(staging, destination)

            # Reopen by immutable identity before changing the shared pointer.
            # This also catches a corrupt pre-existing object or generation
            # with the same content identity.
            snapshot = self.open_generation(generation_id)
            self.root.mkdir(parents=True, exist_ok=True)
            temporary_pointer = self.root / f".current-{uuid.uuid4().hex}.json"
            temporary_pointer.write_text(
                json.dumps({"generation_id": generation_id}), encoding="utf-8"
            )
            os.replace(temporary_pointer, self.current)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return snapshot

    def open_current(self) -> ExpectedLibrarySnapshot:
        if not self.current.is_file():
            raise ExpectedLibraryNotBuilt("Expected library has not been built")
        try:
            pointer = json.loads(self.current.read_text(encoding="utf-8"))
            generation_id = pointer["generation_id"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ExpectedLibraryCorrupt(
                "Expected library current pointer is invalid"
            ) from exc
        return self.open_generation(str(generation_id))

    def open_generation(self, generation_id: str) -> ExpectedLibrarySnapshot:
        try:
            if (
                len(generation_id) != 64
                or any(character not in "0123456789abcdef" for character in generation_id)
            ):
                raise ValueError("invalid generation identity")
            return self._open_generation(generation_id)
        except ExpectedLibraryCorrupt:
            raise
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExpectedLibraryCorrupt(
                f"Expected library generation {generation_id!r} manifest is invalid"
            ) from exc

    def _open_generation(self, generation_id: str) -> ExpectedLibrarySnapshot:
        generation_dir = self.generations / generation_id
        manifest = json.loads(
            (generation_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("generation_id") != generation_id:
            raise ValueError("Expected library manifest identity mismatch")
        if _generation_identity(manifest["documents"]) != generation_id:
            raise ExpectedLibraryCorrupt("Expected library generation identity mismatch")
        documents = []
        for record in manifest["documents"]:
            document_id = record["document_id"]
            validate_library_segment(document_id, "document identity")
            index_path = generation_dir / f"{document_id}.json"
            self._verify_object(
                index_path, record["index_sha256"], "document index"
            )
            index_json = index_path.read_text(encoding="utf-8")
            deserialize_document(index_json)
            markdown_path = generation_dir / f"{document_id}.md"
            self._verify_object(
                markdown_path,
                record["markdown_sha256"],
                "canonical Markdown",
            )
            canonical_markdown = markdown_path.read_text(encoding="utf-8")
            source_record = record.get("source")
            source = None
            if source_record is not None:
                sha256 = source_record["sha256"]
                pdf_path = self.objects / "pdf" / f"{sha256}.pdf"
                self._verify_object(pdf_path, sha256, "source PDF")
                ocr_scale = _normalize_ocr_scale(source_record["ocr_scale"])
                assets_list = []
                for asset in source_record.get("assets", []):
                    validate_library_segment(
                        asset["asset_id"], "source asset identity"
                    )
                    asset_path = self.objects / "assets" / asset["sha256"]
                    self._verify_object(
                        asset_path,
                        asset["sha256"],
                        f"source asset {asset['asset_id']!r}",
                    )
                    assets_list.append(SourceAsset(
                        asset_id=asset["asset_id"],
                        filename=asset["filename"],
                        sha256=asset["sha256"],
                        path=asset_path,
                        media_type=asset["media_type"],
                        href=(
                            f"/api/library/{generation_id}/documents/"
                            f"{quote(document_id, safe='')}/assets/"
                            f"{quote(asset['asset_id'], safe='')}"
                        ),
                    ))
                assets = tuple(assets_list)
                source = SourceDocument(
                    sha256=sha256,
                    pdf_path=pdf_path,
                    ocr_scale=ocr_scale,
                    href=(
                        f"/api/library/{generation_id}/documents/"
                        f"{quote(document_id, safe='')}/pdf"
                    ),
                    assets=assets,
                )
            documents.append(LibraryDocument(
                document_id=document_id,
                index_path=index_path,
                canonical_markdown=canonical_markdown,
                source=source,
            ))
        return ExpectedLibrarySnapshot(generation_id, tuple(documents))

    @staticmethod
    def _verify_object(path: Path, expected_sha256: str, label: str) -> None:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ExpectedLibraryCorrupt(f"{label} is missing") from exc
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ExpectedLibraryCorrupt(f"{label} digest mismatch")


def _walk_nodes(nodes: tuple[PageNode, ...] | list[PageNode]):
    for node in nodes:
        yield node
        yield from _walk_nodes(node.children)


def write_source_candidates(
    path: Path, candidates: Mapping[str, SourceCandidate]
) -> None:
    """Write ingest candidate metadata with module-computed content digests."""
    records = {}
    for document_id, candidate in sorted(candidates.items()):
        pdf_sha256 = hashlib.sha256(candidate.pdf_path.read_bytes()).hexdigest()
        records[document_id] = {
            "pdf_path": str(candidate.pdf_path.resolve()),
            "pdf_sha256": pdf_sha256,
            "ocr_scale": candidate.ocr_scale,
            "assets": [{
                "asset_id": asset.asset_id,
                "path": str(asset.path.resolve()),
                "sha256": hashlib.sha256(asset.path.read_bytes()).hexdigest(),
                "media_type": asset.media_type,
            } for asset in candidate.assets],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps({"version": 2, "documents": records}, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_source_candidates(path: Path) -> dict[str, SourceCandidate]:
    """Restore candidate metadata emitted by ``write_source_candidates``."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 2 or not isinstance(
            payload.get("documents"), dict
        ):
            raise ValueError("source candidate manifest must be version 2")
        result = {}
        for document_id, record in payload["documents"].items():
            assets = []
            for raw_asset in record.get("assets", []):
                asset = SourceAssetCandidate(
                    asset_id=raw_asset["asset_id"],
                    path=Path(raw_asset["path"]),
                    media_type=raw_asset["media_type"],
                )
                object.__setattr__(asset, "_expected_sha256", raw_asset["sha256"])
                assets.append(asset)
            source = SourceCandidate(
                pdf_path=Path(record["pdf_path"]),
                ocr_scale=float(record["ocr_scale"]),
                assets=tuple(assets),
            )
            object.__setattr__(source, "_expected_sha256", record["pdf_sha256"])
            result[document_id] = source
        return result
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid source candidate manifest") from exc
