"""Typed provenance from canonical PageIndex text to source-PDF regions.

PDF ingest emits versioned JSON in fenced ``pin`` blocks. PageIndex validates
and lifts those blocks onto nodes while keeping them out of model-visible text.
Visual locations are returned only for a unique verbatim quote or a direct asset
crop; uncertainty produces an explicit unavailable result, never a guess.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

PIN_FENCE_OPEN = "```pin"
PIN_BLOCK_RE = re.compile(r"^```pin\s*$\n(.*?)^```\s*$\n?", re.MULTILINE | re.DOTALL)
IMAGE_LINE_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)[ \t]*$", re.MULTILINE)


class PinValidationError(ValueError):
    """A provenance pin cannot truthfully identify a source location."""


@dataclass(frozen=True)
class PixelBox:
    """Ordered corners in the rendered PDF coordinates used during ingest."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        values = (self.x0, self.y0, self.x1, self.y1)
        if not all(math.isfinite(value) for value in values):
            raise PinValidationError("pixel-box coordinates must be finite")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise PinValidationError("pixel-box corners must define a positive area")

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass(frozen=True)
class SourceSpan:
    """A character interval in canonical PageIndex content and its PDF box."""

    page: int
    start: int
    end: int
    box: PixelBox

    def __post_init__(self) -> None:
        if self.page < 1:
            raise PinValidationError("source-span pages are one-based")
        if self.start < 0 or self.end <= self.start:
            raise PinValidationError("source spans must have a positive character extent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "start": self.start,
            "end": self.end,
            "box": self.box.to_list(),
        }


@dataclass(frozen=True)
class AssetProvenance:
    """Identity and crop URL for a figure or table source."""

    asset_id: str
    asset_type: Literal["figure", "table"]
    image: str

    def __post_init__(self) -> None:
        if self.asset_type not in ("figure", "table"):
            raise PinValidationError("asset type must be figure or table")
        if not self.asset_id.strip() or not self.image.strip():
            raise PinValidationError("asset identity and image are required")

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.asset_id,
            "type": self.asset_type,
            "image": self.image,
        }


@dataclass(frozen=True)
class ProvenancePin:
    """Versioned identity and exact source spans for one PageIndex node."""

    version: int
    document: str
    node_id: str
    spans: tuple[SourceSpan, ...]
    scale: float
    asset: AssetProvenance | None = None

    def __post_init__(self) -> None:
        if self.version != 2:
            raise PinValidationError("only provenance pin version 2 is supported")
        if not self.document.strip() or not self.node_id.strip():
            raise PinValidationError("document and node identity are required")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise PinValidationError("ingest scale must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        data = {
            "version": self.version,
            "document": self.document,
            "nodeId": self.node_id,
            "spans": [span.to_dict() for span in self.spans],
            "scale": self.scale,
        }
        if self.asset is not None:
            data["asset"] = self.asset.to_dict()
        return data


@dataclass(frozen=True)
class NormalizedRegion:
    """A browser-overlay region expressed as fractions of one PDF page."""

    page: int
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class VisualLocation:
    """An exact visual citation, or an explicit absence of one."""

    status: Literal["exact", "unavailable"]
    regions: tuple[NormalizedRegion, ...] = ()
    reason: str | None = None


def emit_pin(pin: ProvenancePin) -> str:
    """Emit the complete public pin representation embedded in markdown."""
    body = json.dumps(pin.to_dict(), ensure_ascii=False, separators=(",", ":"))
    return f"```pin\n{body}\n```\n"


def parse_pin(block: str) -> ProvenancePin:
    """Parse and validate a v2 fenced pin or its JSON body."""
    match = PIN_BLOCK_RE.fullmatch(block)
    body = match.group(1) if match else block
    try:
        raw = json.loads(body)
        if not isinstance(raw, dict):
            raise PinValidationError("provenance pin must be an object")
        if type(raw.get("version")) is not int or raw["version"] != 2:
            raise PinValidationError("provenance pin version must be integer 2")
        if not isinstance(raw.get("document"), str):
            raise PinValidationError("pin document must be text")
        if not isinstance(raw.get("nodeId"), str):
            raise PinValidationError("pin node identity must be text")
        if not isinstance(raw.get("spans"), list):
            raise PinValidationError("pin spans must be a list")
        if isinstance(raw.get("scale"), bool) or not isinstance(
            raw.get("scale"), (int, float)
        ):
            raise PinValidationError("pin scale must be numeric")
        for span in raw["spans"]:
            if not isinstance(span, dict):
                raise PinValidationError("source span must be an object")
            for field in ("page", "start", "end"):
                if type(span.get(field)) is not int:
                    raise PinValidationError(
                        f"source span {field} must be an integer"
                    )
            box = span.get("box")
            if not isinstance(box, list) or len(box) != 4 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in box
            ):
                raise PinValidationError("source span box must contain four numbers")
        asset = raw.get("asset")
        if asset is not None and (
            not isinstance(asset, dict)
            or not isinstance(asset.get("id"), str)
            or not isinstance(asset.get("type"), str)
            or not isinstance(asset.get("image"), str)
        ):
            raise PinValidationError("pin asset must have textual id, type, and image")
        spans = tuple(
            SourceSpan(
                page=span["page"],
                start=span["start"],
                end=span["end"],
                box=PixelBox(*(float(value) for value in span["box"])),
            )
            for span in raw["spans"]
        )
        return ProvenancePin(
            version=raw["version"],
            document=raw["document"],
            node_id=raw["nodeId"],
            spans=spans,
            scale=float(raw["scale"]),
            asset=(AssetProvenance(
                asset_id=raw["asset"]["id"],
                asset_type=raw["asset"]["type"],
                image=raw["asset"]["image"],
            ) if raw.get("asset") is not None else None),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, PinValidationError):
            raise
        raise PinValidationError("invalid provenance pin v2") from exc


def pin_from_dict(raw: dict[str, Any]) -> ProvenancePin:
    """Restore a validated pin from the deterministic index representation."""
    return parse_pin(json.dumps(raw, ensure_ascii=False))


def locate_visual_citation(
    pin: ProvenancePin,
    content: str,
    quote: str,
    page_size: Callable[[str, int], tuple[float, float] | None],
) -> VisualLocation:
    """Locate one verbatim quote, returning no highlight unless it is exact."""
    if pin.asset is not None:
        spans = pin.spans
        if not spans:
            return VisualLocation("unavailable", reason="asset_has_no_source_span")
    else:
        if not quote:
            return VisualLocation("unavailable", reason="missing_quote")
        start = content.find(quote)
        if start < 0:
            return VisualLocation("unavailable", reason="quote_not_found")
        if content.find(quote, start + 1) >= 0:
            return VisualLocation("unavailable", reason="ambiguous_quote")
        end = start + len(quote)
        spans = tuple(span for span in pin.spans if span.start < end and span.end > start)
        if not spans:
            return VisualLocation("unavailable", reason="quote_has_no_source_span")

    if any(span.start < 0 or span.end > len(content) for span in spans):
        return VisualLocation("unavailable", reason="invalid_source_span")
    if pin.asset is None:
        cursor = start
        for span in sorted(spans, key=lambda item: (item.start, item.end)):
            mapped_start = max(start, span.start)
            mapped_end = min(end, span.end)
            if mapped_start > cursor and content[cursor:mapped_start].strip():
                return VisualLocation("unavailable", reason="quote_not_fully_mapped")
            cursor = max(cursor, mapped_end)
        if cursor < end and content[cursor:end].strip():
            return VisualLocation("unavailable", reason="quote_not_fully_mapped")

    regions: list[NormalizedRegion] = []
    for span in spans:
        size = page_size(pin.document, span.page)
        if size is None:
            return VisualLocation("unavailable", reason="page_size_unavailable")
        page_width, page_height = size
        if (not math.isfinite(page_width) or not math.isfinite(page_height)
                or page_width <= 0 or page_height <= 0):
            return VisualLocation("unavailable", reason="invalid_page_size")
        left = max(0.0, min(1.0, span.box.x0 / page_width))
        right = max(0.0, min(1.0, span.box.x1 / page_width))
        top = max(0.0, min(1.0, span.box.y0 / page_height))
        bottom = max(0.0, min(1.0, span.box.y1 / page_height))
        if right <= left or bottom <= top:
            return VisualLocation("unavailable", reason="source_region_outside_page")
        regions.append(NormalizedRegion(
            page=span.page,
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        ))
    return VisualLocation("exact", tuple(regions))


def _strip_pins(text: str) -> str:
    """Remove pin blocks and standalone image lines from LLM-visible content
    (the viewer still renders them from the raw markdown)."""
    text = PIN_BLOCK_RE.sub("", text)
    text = IMAGE_LINE_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_provenance_markup(text: str) -> str:
    """Return the canonical PageIndex content with source metadata removed."""
    return _strip_pins(text)
