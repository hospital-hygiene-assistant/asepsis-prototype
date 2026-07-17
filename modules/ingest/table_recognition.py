"""Focused, fail-closed table recognition for confirmed review crops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TableRecognitionUnavailable(RuntimeError):
    """The focused Paddle table pipeline could not produce structured output."""


@dataclass(frozen=True)
class TableRecognition:
    markdown: str
    payload: dict[str, Any]


_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from paddleocr import TableRecognitionPipelineV2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise TableRecognitionUnavailable(
                "Table recognition requires the locked OCR dependency group"
            ) from exc
        _pipeline = TableRecognitionPipelineV2(
            device="cpu",
            text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=False,
            use_ocr_model=True,
        )
    return _pipeline


def _plain_payload(value: Any) -> dict[str, Any]:
    raw = getattr(value, "json", value)
    if callable(raw):
        raw = raw()
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise TableRecognitionUnavailable("Paddle returned no structured table result")
    return raw


def _find_markdown(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if "|" in text or text.startswith("<table"):
            return text
        return ""
    if isinstance(value, dict):
        for key in ("markdown", "pred_markdown", "table_markdown", "pred_html", "html"):
            hit = _find_markdown(value.get(key))
            if hit:
                return hit
        for nested in value.values():
            hit = _find_markdown(nested)
            if hit:
                return hit
    if isinstance(value, (list, tuple)):
        for nested in value:
            hit = _find_markdown(nested)
            if hit:
                return hit
    return ""


def recognize_table(image_path: Path) -> TableRecognition:
    """Recognise one confirmed crop; never repair or invent missing output."""
    try:
        results = list(_load_pipeline().predict(input=str(image_path)))
        if not results:
            raise TableRecognitionUnavailable("Paddle returned no table result")
        payload = _plain_payload(results[0])
        markdown = _find_markdown(payload)
        if not markdown:
            raise TableRecognitionUnavailable("Paddle returned no table structure")
        return TableRecognition(markdown=markdown, payload=payload)
    except TableRecognitionUnavailable:
        raise
    except Exception as exc:
        raise TableRecognitionUnavailable(str(exc) or type(exc).__name__) from exc
