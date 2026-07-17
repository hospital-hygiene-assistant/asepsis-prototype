"""
ingest/ocr.py — fast structure OCR: layout detection + lite title recognition.

The full PaddleOCR-VL 0.9B model runs ~50-100s PER PAGE on this CPU-only machine,
dominated by autoregressive text generation over every block.  For the heading
TREE we don't need that: PP-DocLayoutV3 labels every block (title/text/figure)
in ~1s/page with no generation, and a mobile PP-OCR model reads the few short
title blocks in a fraction of a second — at identical heading quality.

So: layout-detect each page, then recognise ONLY the title blocks (configurable).
~40-60x faster than the VLM for structure.  Results cached by (pdf, config).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_DEFAULT_CACHE_DIR = Path(".ocr_cache")

# lazily-loaded singletons (heavy) — reused across documents in one process
_layout = None
_recognizer = None


@dataclass
class Block:
    label: str   # "doc_title" | "paragraph_title" | "text" | "table" | "figure" | …
    text: str    # recognised text (empty for blocks we didn't recognise)
    page: int
    bbox: tuple[float, float, float, float]   # rendered-image pixels


@dataclass(frozen=True)
class OcrConfig:
    ocr_scale: float = 2.0        # PDF→image render scale (144 DPI at 2.0)
    titles_only: bool = True      # recognise only title blocks (fast structure path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(pdf_path: Path, config: OcrConfig, cache_dir: Path) -> Path:
    tag = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:8]
    return cache_dir / f"{_sha256(pdf_path)}.layout.{tag}.json"


def _p(msg: str) -> None:
    print(f"[OCR] {msg}", flush=True)


def _load_models():
    global _layout, _recognizer
    if _layout is None:
        from paddlex import create_model  # type: ignore[import-untyped]
        _layout = create_model("PP-DocLayoutV3")
    if _recognizer is None:
        from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        _recognizer = PaddleOCR(
            use_textline_orientation=False, lang="latin",
            text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
            device="cpu",
            use_doc_orientation_classify=False, use_doc_unwarping=False)
    return _layout, _recognizer


def _recognise(recognizer, pil_crop) -> str:
    import numpy as np
    r = recognizer.predict(np.array(pil_crop))
    if r and r[0].get("rec_texts"):
        return " ".join(r[0]["rec_texts"]).strip()
    return ""


def run_ocr(
    pdf_path: Path,
    config: OcrConfig = OcrConfig(),
    *,
    cache_dir: Path | None = _DEFAULT_CACHE_DIR,
) -> list[Block]:
    """Layout-detect every page; recognise title blocks.  Cached by (pdf, config)."""
    if cache_dir is not None:
        cp = _cache_path(pdf_path, config, cache_dir)
        if cp.exists():
            _p(f"cache hit — {cp.name}")
            return [Block(label=b["label"], text=b["text"], page=b["page"],
                          bbox=tuple(b["bbox"])) for b in json.loads(cp.read_text())]

    import numpy as np
    import pypdfium2 as pdfium  # type: ignore[import-untyped]

    _p(f"OCR {pdf_path.name} (layout + lite titles, config={config}) …")
    layout, recognizer = _load_models()
    doc = pdfium.PdfDocument(str(pdf_path))

    blocks: list[Block] = []
    t0 = time.time()
    for page_idx in range(len(doc)):
        img = doc[page_idx].render(scale=config.ocr_scale).to_pil()
        boxes = list(layout.predict(np.array(img)))[0]["boxes"]
        for b in boxes:
            label = b["label"]
            x0, y0, x1, y1 = (float(v) for v in b["coordinate"])
            text = ""
            recognise = (not config.titles_only) or label.endswith("title")
            if recognise:
                text = _recognise(recognizer, img.crop((int(x0), int(y0), int(x1), int(y1))))
            blocks.append(Block(label=label, text=text, page=page_idx, bbox=(x0, y0, x1, y1)))
        _p(f"page {page_idx + 1}/{len(doc)} done ({time.time() - t0:.0f}s)")

    _p(f"extracted {len(blocks)} blocks in {time.time() - t0:.0f}s")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = _cache_path(pdf_path, config, cache_dir)
        cp.write_text(json.dumps(
            [{"label": b.label, "text": b.text, "page": b.page, "bbox": list(b.bbox)}
             for b in blocks], indent=2, ensure_ascii=False))
        _p(f"cached → {cp.name}")

    return blocks
