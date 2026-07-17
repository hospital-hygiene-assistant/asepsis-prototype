"""Bringing documents into the corpus."""

import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pageindex

from modules.registry import load as _load_module
from paths import KB_DIR
from .reviews import review_store

router = APIRouter()


# ---------------------------------------------------------------------------

_ingest_lock = threading.Lock()
_ingest_state: dict = {"state": "idle", "phase": "", "doc": "", "done": 0,
                       "total": 0, "message": "", "warnings": [], "docs": [],
                       "review_id": None}


def _set_ingest_state(**kw) -> None:
    with _ingest_lock:
        _ingest_state.update(kw)


def _rebuild_index() -> None:
    documents = [path.stem for path in sorted(KB_DIR.glob("*.md"))]
    pageindex.build_generation(documents)


@router.get("/api/ingest/source")
def get_ingest_source(module: str = "betteringest_pdf"):
    """Whether the given ingest module needs a source folder, and the one
    currently configured (if any)."""
    try:
        mod = _load_module("ingest", module)
    except Exception as exc:
        return JSONResponse({"error": f"unknown ingest module '{module}': {exc}"},
                            status_code=404)
    info = getattr(mod, "MODULE_INFO", {})
    source_dir = mod.get_source_dir() if hasattr(mod, "get_source_dir") else None
    return JSONResponse({
        "module": module,
        "needs_source": info.get("source") == "pdf_folder",
        "source_dir": source_dir,
        "source_ok": bool(source_dir and Path(source_dir).is_dir()),
    })


class IngestSourceRequest(BaseModel):
    module: str = "betteringest_pdf"
    path: str


@router.post("/api/ingest/source")
def post_ingest_source(req: IngestSourceRequest):
    """Validate and persist the PDF source folder for a folder-based module."""
    try:
        mod = _load_module("ingest", req.module)
    except Exception as exc:
        return JSONResponse({"error": f"unknown ingest module '{req.module}': {exc}"},
                            status_code=404)
    folder = Path(req.path).expanduser()
    if not folder.is_dir():
        return JSONResponse({"error": f"Not a folder: {folder}"}, status_code=400)
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        return JSONResponse({"error": f"No .pdf files in {folder}"}, status_code=400)
    if not hasattr(mod, "set_source_dir"):
        return JSONResponse({"error": f"module '{req.module}' takes no source folder"},
                            status_code=400)
    mod.set_source_dir(str(folder))
    return JSONResponse({"source_dir": str(folder), "pdf_count": len(pdfs),
                         "pdfs": [p.name for p in pdfs]})


class IngestRunRequest(BaseModel):
    ingest_module: str = "betteringest_pdf"
    source_dir: Optional[str] = None


@router.post("/api/ingest/run")
def run_ingest(req: IngestRunRequest):
    """Run ingest (then re-index the knowledge base) in a background thread;
    the frontend polls /api/ingest/progress."""
    with _ingest_lock:
        if _ingest_state["state"] == "running":
            return JSONResponse({"error": "an ingest run is already in progress"},
                                status_code=409)
        _ingest_state.update({"state": "running", "phase": "starting", "doc": "",
                              "done": 0, "total": 0, "message": "Starting…",
                              "warnings": [], "docs": [], "review_id": None})

    ingest_name = req.ingest_module
    source_dir = req.source_dir

    def work():
        try:
            ingest_mod = _load_module("ingest", ingest_name)
            info = getattr(ingest_mod, "MODULE_INFO", {})
            if info.get("review_required") is True:
                documents = ingest_mod.prepare_review(
                    source_dir=source_dir,
                    progress=lambda progress: _set_ingest_state(**progress),
                )
                session = review_store.create(documents)
                _set_ingest_state(
                    state="review",
                    phase="review",
                    message="Review figures and tables before publishing",
                    docs=[document.document_id for document in session.documents],
                    review_id=session.session_id,
                )
                return
            kwargs = {}
            if info.get("source") == "pdf_folder":
                kwargs = {"source_dir": source_dir, "progress":
                          lambda info: _set_ingest_state(**info)}
            result = ingest_mod.run(**kwargs) or {}

            _set_ingest_state(phase="index", message="Rebuilding index…")
            _rebuild_index()

            _set_ingest_state(state="done", phase="done",
                              message="Ingest + index complete",
                              warnings=result.get("warnings", []),
                              docs=result.get("docs", []))
        except Exception as exc:
            _set_ingest_state(state="error", message=str(exc))

    threading.Thread(target=work, daemon=True).start()
    return JSONResponse({"started": True})


class IngestAddRequest(BaseModel):
    path: str
    ingest_module: str = "betteringest_pdf"


@router.post("/api/ingest/add")
def add_to_library(req: IngestAddRequest):
    """Additive ingest: bring a single PDF or a folder of PDFs into the
    library on top of the existing corpus, then re-index. Existing documents
    are untouched (same-named stems are refreshed)."""
    p = Path(req.path).expanduser()
    if p.is_file() and p.suffix.lower() == ".pdf":
        pdfs = [p]
    elif p.is_dir():
        pdfs = sorted(p.glob("*.pdf"))
    else:
        return JSONResponse({"error": f"Not a .pdf file or a folder: {p}"},
                            status_code=400)
    if not pdfs:
        return JSONResponse({"error": f"No .pdf files in {p}"}, status_code=400)

    try:
        ingest_mod = _load_module("ingest", req.ingest_module)
    except Exception as exc:
        return JSONResponse({"error": f"unknown ingest module '{req.ingest_module}': {exc}"},
                            status_code=404)
    if not hasattr(ingest_mod, "run_paths"):
        return JSONResponse({"error": f"module '{req.ingest_module}' does not support additive ingest"},
                            status_code=400)

    with _ingest_lock:
        if _ingest_state["state"] == "running":
            return JSONResponse({"error": "an ingest run is already in progress"},
                                status_code=409)
        _ingest_state.update({"state": "running", "phase": "starting", "doc": "",
                              "done": 0, "total": len(pdfs), "message": "Starting…",
                              "warnings": [], "docs": [], "review_id": None})

    def work():
        try:
            if getattr(ingest_mod, "MODULE_INFO", {}).get("review_required") is True:
                documents = ingest_mod.prepare_review_paths(
                    [str(x) for x in pdfs],
                    progress=lambda progress: _set_ingest_state(**progress),
                )
                session = review_store.create(documents)
                _set_ingest_state(
                    state="review",
                    phase="review",
                    message="Review figures and tables before publishing",
                    docs=[document.document_id for document in session.documents],
                    review_id=session.session_id,
                )
                return
            result = ingest_mod.run_paths(
                [str(x) for x in pdfs],
                progress=lambda info: _set_ingest_state(**info)) or {}

            _set_ingest_state(phase="index", message="Rebuilding index…")
            _rebuild_index()

            _set_ingest_state(state="done", phase="done",
                              message=f"Added {len(result.get('docs', []))} document(s)",
                              warnings=result.get("warnings", []),
                              docs=result.get("docs", []))
        except Exception as exc:
            _set_ingest_state(state="error", message=str(exc))

    threading.Thread(target=work, daemon=True).start()
    return JSONResponse({"started": True, "pdf_count": len(pdfs),
                         "pdfs": [x.name for x in pdfs]})


@router.get("/api/ingest/progress")
def ingest_progress():
    with _ingest_lock:
        return JSONResponse(dict(_ingest_state))
