"""
Vendored BetterIngester PDF-ingestion pipeline (deterministic, local,
structure-faithful PDF → markdown + managed assets).

Scope: only ingest/betteringest.py and its runtime dependencies (ladder,
levels, outline, ocr, reconstruct) plus the minimal shared tree module and the
deterministic page-anchor helpers.  BetterIngester's benchmark/scoring study
(run_pipeline.py, bench scoring, tools/, experiments/) is intentionally not
vendored.

Upstream: /Users/fede/Desktop/VSCODEProjects/BetterIngester (see each file's
docstring for the per-file deviations, all of which are astepsis-specific
plumbing rather than algorithm changes).
"""
from .betteringest import Asset, BetterIngest, IngestedDoc
from .ocr import Block, OcrConfig, run_ocr

__all__ = ["Asset", "BetterIngest", "IngestedDoc", "Block", "OcrConfig",
           "run_ocr"]
