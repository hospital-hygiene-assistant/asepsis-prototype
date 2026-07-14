"""Asepsis Prototype — FastAPI backend. Serves the UI and wraps pageindex retrieval."""

import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Add project root so we can import pageindex and modules
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pageindex as _pi
from pageindex import _node_from_dict
from paths import ASSETS_DIR
from modules.registry import discover as _discover_modules, load as _load_module, defaults as _module_defaults

import re

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from api.pdf import _normalized_bbox
from api.sources import load_sources
from api.prompts import (
    CHAT_SYNTHESIS_PROMPT,
    _context_block,
    _parse_answer_sections,
)
from api.ollama_pool import (
    BASE_PORT,
    ensure_explainer,
    ollama_bin,
    set_ollama_instances,
    shutdown_pool,
    start_configured_instances,
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# The vanilla dev console is served from this process by default, so the Tauri
# launcher keeps working untouched. ASEPSIS_SERVE_UI=0 runs a bare API, which is
# what a separately-hosted frontend needs.
SERVE_UI = _env_flag("ASEPSIS_SERVE_UI", True)

# A same-origin deployment needs no CORS at all; the default here only covers the
# Next.js dev server. Comma-separated list, or "*" to allow any origin.
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ASEPSIS_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bring the Ollama pool up on startup and tear the extra instances down.

    Importing this module must stay side-effect free so tests can load it
    without spawning processes.
    """
    start_configured_instances()
    yield
    shutdown_pool()


app = FastAPI(title="Asepsis Prototype", lifespan=lifespan)
if CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                       allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def _no_cache(request, call_next):
    """Prevent the (WK)webview from serving stale UI assets.

    Tauri loads the UI from this server's devUrl and does no hot-reload, so the
    webview will otherwise cache main.js/index.html across launches and ignore
    edits. Disabling caching keeps every launch on the latest UI.
    """
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


UI_DIR = Path(__file__).parent / "ui"
# check_dir=False so a missing or removed ui/ degrades to 404s on the console
# routes instead of taking the whole API down at import time.
if SERVE_UI:
    app.mount("/static", StaticFiles(directory=UI_DIR, check_dir=False), name="static")

# check_dir=False: the crops directory only appears on the first PDF ingest.
app.mount("/assets", StaticFiles(directory=ASSETS_DIR, check_dir=False), name="assets")



# ---------------------------------------------------------------------------
# Test definitions — mirrors tests/test_retrieval.py exactly
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": "test_sodium_restriction",
        "category": "SINGLE",
        "description": "Sodium restriction",
        "query": "What sodium intake level is recommended for hypertension and by how much does it reduce blood pressure?",
        "expected": {"hypertension_guidelines": ["sodium-restriction"]},
        "expected_any": {},
    },
    {
        "id": "test_nmba_icu_two_leaves",
        "category": "SINGLE",
        "description": "NMBA in ARDS — two leaves",
        "query": "When are neuromuscular blocking agents indicated in ARDS patients and how is the depth of blockade monitored?",
        "expected": {"icu_sedation_guide": ["indications-in-ards", "monitoring-and-safety"]},
        "expected_any": {},
    },
    {
        "id": "test_hypertension_lifestyle_and_drugs",
        "category": "MULTI",
        "description": "Lifestyle + drug classes",
        "query": "What lifestyle changes and which drug classes should be started for newly diagnosed hypertension?",
        "expected": {"hypertension_guidelines": ["first-line-drug-classes"]},
        "expected_any": {
            "hypertension_guidelines": [
                "sodium-restriction", "dash-diet",
                "exercise-and-weight-management", "non-pharmacological-management-overview",
            ],
        },
    },
    {
        "id": "test_sepsis_antibiotics_empiric_and_deescalation",
        "category": "MULTI",
        "description": "Sepsis antibiotics — empiric + de-escalation",
        "query": "How should empiric antibiotics be chosen for sepsis by source of infection, and when should they be narrowed?",
        "expected": {"antibiotic_stewardship": ["empiric-regimens-by-source", "de-escalation-and-duration"]},
        "expected_any": {},
    },
    {
        "id": "test_hypertension_ckd_cross_doc",
        "category": "CROSS",
        "description": "CKD antihypertensives + renal screening",
        "query": "What antihypertensives are preferred for patients with CKD and what renal complications should be monitored?",
        "expected": {
            "hypertension_guidelines": ["hypertension-in-ckd"],
            "diabetes_management": ["complication-screening"],
        },
        "expected_any": {},
    },
    {
        "id": "test_septic_icu_patient",
        "category": "CROSS",
        "description": "Septic ICU patient — antibiotics + sedation",
        "query": "A patient with septic shock is intubated in the ICU — what empiric antibiotics and sedation agents should be used?",
        "expected": {"antibiotic_stewardship": ["empiric-regimens-by-source"]},
        "expected_any": {"icu_sedation_guide": ["opioids", "propofol", "dexmedetomidine"]},
    },
]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _versioned(path: str) -> str:
    """Append a cache-busting ?v=<mtime> to a /static/<file> reference.

    Tauri's WKWebView keeps an on-disk asset cache that survives app quit and
    can serve a stale main.js/index.html even with no-store headers. Versioning
    the URL by file mtime forces a fresh fetch whenever a UI file changes.
    """
    fname = path.rsplit("/static/", 1)[-1]
    fpath = UI_DIR / fname
    try:
        return f"{path}?v={int(fpath.stat().st_mtime)}"
    except OSError:
        return path


@app.get("/")
def root():
    if not SERVE_UI:
        return JSONResponse(
            {"service": "asepsis-prototype", "ui": "disabled", "api": "/api"})
    index = UI_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"error": "dev console is not installed"}, status_code=404)
    html = re.sub(r"/static/[\w.\-]+", lambda m: _versioned(m.group(0)),
                  index.read_text(encoding="utf-8"))
    return HTMLResponse(html)


@app.get("/api/tests")
def get_tests():
    return JSONResponse(TEST_CASES)


@app.get("/api/status")
def get_status():
    activity = _pi.get_activity()
    with _chat_lock:
        chat_phase = dict(_chat_phase)
    return JSONResponse({
        "instances": [
            {"url": url, "index": i, "active": activity.get(url, 0)}
            for i, url in enumerate(_pi.OLLAMA_URLS)
        ],
        "any_busy": any(v > 0 for v in activity.values()),
        "progress": _pi.get_progress(),
        "live":     _pi.get_live_events(),
        "chat":     chat_phase,
    })


@app.get("/api/config")
def get_config():
    return JSONResponse({
        "ollama_instances": len(_pi.OLLAMA_URLS),
        "ollama_urls": _pi.OLLAMA_URLS,
        "ollama_bin_available": ollama_bin() is not None,
        "retrieval_model": _pi.MODEL,
        "synthesis_model": getattr(_pi, "SYNTHESIS_MODEL", _pi.MODEL),
    })


class ConfigRequest(BaseModel):
    ollama_instances: int
    retrieval_model: Optional[str] = None
    synthesis_model: Optional[str] = None


@app.post("/api/config")
def post_config(req: ConfigRequest):
    result = set_ollama_instances(req.ollama_instances)
    if req.retrieval_model:
        _pi.MODEL = req.retrieval_model
    if req.synthesis_model:
        _pi.SYNTHESIS_MODEL = req.synthesis_model
    result["retrieval_model"] = _pi.MODEL
    result["synthesis_model"] = getattr(_pi, "SYNTHESIS_MODEL", _pi.MODEL)
    return JSONResponse(result)


@app.get("/api/models")
def list_models():
    try:
        client = _pi.make_client(_pi.OLLAMA_URLS[0])
        res = client.list()
        models = [m["name"] for m in res.get("models", [])]
        return JSONResponse({"models": models})
    except Exception as e:
        return JSONResponse({"error": str(e), "models": []})


@app.get("/api/modules")
def get_modules():
    registry = _discover_modules()
    defs = _module_defaults()
    return JSONResponse({
        "stages": {
            stage: {
                "default": defs.get(stage),
                "modules": list(mods.values()),
            }
            for stage, mods in registry.items()
        }
    })


@app.get("/api/documents")
def get_documents():
    # Use the default index module to find where indices live
    index_mod = _load_module("index", _module_defaults()["index"])
    index_dir = getattr(index_mod, "INDEX_DIR", Path("index"))
    docs = []
    for idx in sorted(index_dir.glob("*.json")):
        tree = json.loads(idx.read_text(encoding="utf-8"))
        nodes = [_node_from_dict(d) for d in tree]
        docs.append({
            "name": idx.stem,
            "leaf_count": _count_leaves(nodes),
            "tree": tree,
        })
    return JSONResponse(docs)


@app.get("/api/document/{stem}/full")
def get_document_full(stem: str):
    """Return the raw markdown for a single knowledge-base document.

    The frontend renders the markdown and derives its own table of contents
    from the rendered headings, so anchors always stay consistent.
    """
    # Resolve the KB directory from the default ingest module, relative to ROOT.
    try:
        ingest_mod = _load_module("ingest", _module_defaults()["ingest"])
        kb_dir = getattr(ingest_mod, "KB_DIR", Path("knowledge_base"))
    except Exception:
        kb_dir = Path("knowledge_base")
    if not kb_dir.is_absolute():
        kb_dir = ROOT / kb_dir

    # Guard against path traversal — only allow plain stems that exist.
    safe_stem = Path(stem).name
    md_path = kb_dir / f"{safe_stem}.md"
    if not md_path.exists() or md_path.parent.resolve() != kb_dir.resolve():
        return JSONResponse({"error": f"Document '{stem}' not found"}, status_code=404)

    return JSONResponse({
        "stem": safe_stem,
        "markdown": md_path.read_text(encoding="utf-8"),
    })


# ---------------------------------------------------------------------------
# BetterIngest PDF ingest — source-folder config, background run, progress
# ---------------------------------------------------------------------------

_ingest_lock = threading.Lock()
_ingest_state: dict = {"state": "idle", "phase": "", "doc": "", "done": 0,
                       "total": 0, "message": "", "warnings": [], "docs": []}


def _set_ingest_state(**kw) -> None:
    with _ingest_lock:
        _ingest_state.update(kw)


@app.get("/api/ingest/source")
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


@app.post("/api/ingest/source")
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
    index_module: Optional[str] = None
    source_dir: Optional[str] = None


@app.post("/api/ingest/run")
def run_ingest(req: IngestRunRequest):
    """Run ingest (then re-index the knowledge base) in a background thread;
    the frontend polls /api/ingest/progress."""
    with _ingest_lock:
        if _ingest_state["state"] == "running":
            return JSONResponse({"error": "an ingest run is already in progress"},
                                status_code=409)
        _ingest_state.update({"state": "running", "phase": "starting", "doc": "",
                              "done": 0, "total": 0, "message": "Starting…",
                              "warnings": [], "docs": []})

    ingest_name = req.ingest_module
    index_name = req.index_module or _module_defaults()["index"]
    source_dir = req.source_dir

    def work():
        try:
            ingest_mod = _load_module("ingest", ingest_name)
            kwargs = {}
            if getattr(ingest_mod, "MODULE_INFO", {}).get("source") == "pdf_folder":
                kwargs = {"source_dir": source_dir, "progress":
                          lambda info: _set_ingest_state(**info)}
            result = ingest_mod.run(**kwargs) or {}

            _set_ingest_state(phase="index", message="Rebuilding index…")
            index_mod = _load_module("index", index_name)
            kb_dir = getattr(ingest_mod, "KB_DIR", Path("knowledge_base"))
            if not Path(kb_dir).is_absolute():
                kb_dir = ROOT / kb_dir
            for md in sorted(Path(kb_dir).glob("*.md")):
                index_mod.build_index(md.stem)

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
    index_module: Optional[str] = None


@app.post("/api/ingest/add")
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
                              "warnings": [], "docs": []})

    index_name = req.index_module or _module_defaults()["index"]

    def work():
        try:
            result = ingest_mod.run_paths(
                [str(x) for x in pdfs],
                progress=lambda info: _set_ingest_state(**info)) or {}

            _set_ingest_state(phase="index", message="Rebuilding index…")
            index_mod = _load_module("index", index_name)
            kb_dir = getattr(ingest_mod, "KB_DIR", Path("knowledge_base"))
            if not Path(kb_dir).is_absolute():
                kb_dir = ROOT / kb_dir
            for md in sorted(Path(kb_dir).glob("*.md")):
                index_mod.build_index(md.stem)

            _set_ingest_state(state="done", phase="done",
                              message=f"Added {len(result.get('docs', []))} document(s)",
                              warnings=result.get("warnings", []),
                              docs=result.get("docs", []))
        except Exception as exc:
            _set_ingest_state(state="error", message=str(exc))

    threading.Thread(target=work, daemon=True).start()
    return JSONResponse({"started": True, "pdf_count": len(pdfs),
                         "pdfs": [x.name for x in pdfs]})


@app.get("/api/ingest/progress")
def ingest_progress():
    with _ingest_lock:
        return JSONResponse(dict(_ingest_state))


# Rendered-page cache — chat citation previews request the same page/bbox
# repeatedly; pdfium renders are ~100ms each, so memoize the PNG bytes.
_page_png_cache: dict = {}
_page_png_lock = threading.Lock()
_PAGE_CACHE_MAX = 64


@app.get("/api/document/{stem}/pdf")
def get_document_pdf(stem: str):
    """Serve a document's source PDF so a client can render it itself.

    The path is resolved from the server-side manifest keyed by a bare stem, so
    `stem` never reaches the filesystem — same guard as the other document routes.
    """
    safe_stem = Path(stem).name
    src = load_sources().get(safe_stem)
    if not src:
        return JSONResponse(
            {"error": f"'{safe_stem}' has no source PDF (not ingested from a PDF)"},
            status_code=404)
    pdf_path = Path(src["pdf"])
    if not pdf_path.exists():
        return JSONResponse({"error": f"source PDF for '{safe_stem}' is unavailable"},
                            status_code=404)
    return FileResponse(pdf_path, media_type="application/pdf",
                        filename=f"{safe_stem}.pdf")


@app.get("/api/document/{stem}/page/{page}")
def get_document_page(stem: str, page: int, bbox: Optional[str] = None,
                      regions: Optional[str] = None):
    """Render one page of the document's SOURCE PDF as a PNG, optionally with
    the pin's bbox highlighted — powers 'jump to its page/bbox' in the UI.

    `bbox` is "x0,y0,x1,y1" and `regions` is JSON [[page,x0,y0,x1,y1],...],
    both in render pixels at the ingest ocr_scale (the pin convention)."""
    src = load_sources().get(Path(stem).name)
    if not src:
        return JSONResponse(
            {"error": f"'{stem}' has no source PDF (not ingested from a PDF)"},
            status_code=404)
    pdf_path = Path(src["pdf"])
    if not pdf_path.exists():
        return JSONResponse({"error": f"source PDF moved or deleted: {pdf_path}"},
                            status_code=404)
    try:
        import pypdfium2 as pdfium
        from PIL import ImageDraw
    except ImportError as exc:
        return JSONResponse(
            {"error": f"page rendering needs pypdfium2 + Pillow: {exc}"},
            status_code=501)

    cache_key = (str(pdf_path), pdf_path.stat().st_mtime, page, bbox or "", regions or "")
    with _page_png_lock:
        cached = _page_png_cache.get(cache_key)
    if cached is not None:
        return Response(content=cached, media_type="image/png")

    scale = float(src.get("ocr_scale", 2.0))
    doc = pdfium.PdfDocument(str(pdf_path))
    if not (1 <= page <= len(doc)):
        return JSONResponse({"error": f"page {page} out of range 1..{len(doc)}"},
                            status_code=404)
    img = doc[page - 1].render(scale=scale).to_pil().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    if regions:
        try:
            for r in json.loads(regions):
                if int(r[0]) == page:
                    draw.rectangle([r[1], r[2], r[3], r[4]],
                                   outline=(56, 189, 248, 220), width=3)
        except Exception:
            pass
    if bbox:
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox.split(","))
            draw.rectangle([x0, y0, x1, y1], fill=(245, 158, 11, 56),
                           outline=(245, 158, 11, 255), width=4)
        except Exception:
            pass

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    with _page_png_lock:
        if len(_page_png_cache) >= _PAGE_CACHE_MAX:
            _page_png_cache.pop(next(iter(_page_png_cache)))
        _page_png_cache[cache_key] = data
    return Response(content=data, media_type="image/png")


class ExplainRequest(BaseModel):
    stem: str
    node_id: str
    query: str


@app.post("/api/explain")
def explain_node(req: ExplainRequest):
    """On-demand grounded explanation for a node that was NOT selected.

    Runs on a dedicated Ollama instance so it never disturbs an in-flight query.
    """
    index_mod = _load_module("index", _module_defaults()["index"])
    index_dir = getattr(index_mod, "INDEX_DIR", Path("index"))
    idx = index_dir / f"{Path(req.stem).name}.json"
    if not idx.exists():
        return JSONResponse({"error": f"unknown document '{req.stem}'"}, status_code=404)

    nodes = [_node_from_dict(d) for d in json.loads(idx.read_text(encoding="utf-8"))]
    nodes_by_id = _pi._build_nodes_by_id(nodes)
    node = nodes_by_id.get(req.node_id)
    if node is None:
        return JSONResponse({"error": f"unknown node '{req.node_id}'"}, status_code=404)

    url = ensure_explainer()
    if not url:
        return JSONResponse({"error": "no Ollama instance available"}, status_code=503)

    parent_map = _pi._build_parent_map(nodes)
    breadcrumb = _pi._make_breadcrumb(req.node_id, parent_map, nodes_by_id)
    client = _pi.make_client(url)
    result = _pi.explain_nonselection(node, req.query, req.stem, breadcrumb, client, url)
    result["node_id"] = req.node_id
    result["instance"] = url
    result["pin"] = node.pin
    return JSONResponse(result)


class RunRequest(BaseModel):
    test_id: Optional[str] = None
    query: Optional[str] = None
    # Module selections — default to pipeline defaults when not supplied
    ingest_module: Optional[str] = None
    index_module:  Optional[str] = None
    query_module:  Optional[str] = None


def _run_retrieval(query: str, index_mod) -> dict:
    """Two-phase retrieval across every indexed document. Shared by /api/run
    (retrieval tab) and /api/chat (chatbot tab) so both drive the same live
    treemap/progress state."""
    index_dir = getattr(index_mod, "INDEX_DIR", Path("index"))

    # Count total leaves across all docs so progress polling has a denominator
    total_leaves = sum(
        _count_leaves([_node_from_dict(d) for d in json.loads(idx.read_text(encoding="utf-8"))])
        for idx in sorted(index_dir.glob("*.json"))
    )
    _pi.start_run(total_leaves)

    results = {}
    for idx in sorted(index_dir.glob("*.json")):
        doc_name = idx.stem
        try:
            retrieve_fn = getattr(index_mod, "retrieve_with_metadata", None)
            if retrieve_fn:
                nodes, node_reasons = retrieve_fn(doc_name, query)
            else:
                nodes = index_mod.retrieve(doc_name, query)
                node_reasons = {}
        except FileNotFoundError:
            continue

        raw_tree = json.loads(idx.read_text(encoding="utf-8"))
        results[doc_name] = {
            "tree": raw_tree,
            "retrieved_ids": [n.node_id for n in nodes],
            "node_meta": node_reasons,  # {node_id: {reason, quote}}
            "nodes": [
                {
                    "node_id": n.node_id,
                    "title": n.title,
                    "content": n.content or "",
                    "synthetic": n.synthetic,
                    "heading_level": n.heading_level,
                    "summary": n.summary,
                    "pin": n.pin,
                    "reason": (node_reasons.get(n.node_id) or {}).get("reason", ""),
                    "quote":  (node_reasons.get(n.node_id) or {}).get("quote",  ""),
                }
                for n in nodes
            ],
        }
    return results


@app.post("/api/run")
def run_query(req: RunRequest):
    test = None
    query = req.query

    if req.test_id:
        test = next((t for t in TEST_CASES if t["id"] == req.test_id), None)
        if test:
            query = test["query"]

    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)

    # Resolve modules
    defs = _module_defaults()
    index_mod_name = req.index_module or defs["index"]
    try:
        index_mod = _load_module("index", index_mod_name)
    except Exception as exc:
        return JSONResponse({"error": f"Could not load index module '{index_mod_name}': {exc}"}, status_code=400)

    results = _run_retrieval(query, index_mod)

    test_result = _eval_test(test, results) if test else None
    return JSONResponse({
        "query": query,
        "results": results,
        "test_result": test_result,
        "pipeline": {
            "ingest": req.ingest_module or _module_defaults()["ingest"],
            "index":  index_mod_name,
            "query":  req.query_module  or _module_defaults()["query"],
        },
    })


# ---------------------------------------------------------------------------
# Chat — retrieval + grounded answer synthesis for the chatbot tab
# ---------------------------------------------------------------------------

_chat_lock = threading.Lock()
_chat_phase: dict = {"phase": "idle", "detail": ""}


def _set_chat_phase(phase: str, detail: str = "") -> None:
    with _chat_lock:
        _chat_phase.update({"phase": phase, "detail": detail})





def _count_eval_errors(results: dict) -> int:
    """Leaves the model could not evaluate, across every document in a run.

    Distinct from rejected leaves: these were never actually checked.
    """
    return sum(
        1
        for doc_data in results.values()
        for meta in (doc_data.get("node_meta") or {}).values()
        if meta.get("status") == "error"
    )


def _breadcrumbs_for(results: dict) -> dict:
    """{doc: {node_id: 'Doc › Section › Leaf'}} for every retrieved node."""
    crumbs: dict[str, dict] = {}
    for doc_name, doc_data in results.items():
        nodes = [_node_from_dict(d) for d in doc_data["tree"]]
        nodes_by_id = _pi._build_nodes_by_id(nodes)
        parent_map = _pi._build_parent_map(nodes)
        crumbs[doc_name] = {
            nid: _pi._make_breadcrumb(nid, parent_map, nodes_by_id)
            for nid in doc_data["retrieved_ids"]
        }
    return crumbs


@app.get("/api/chat/config")
def chat_config():
    """Introspection for the chatbot tab: which model answers, over which
    engine pool, with exactly which prompts — nothing hidden."""
    activity = _pi.get_activity()
    defs = _module_defaults()
    return JSONResponse({
        "model": _pi.MODEL,
        "retrieval_model": _pi.MODEL,
        "synthesis_model": getattr(_pi, "SYNTHESIS_MODEL", _pi.MODEL),
        "temperature": 0,
        "ollama_urls": _pi.OLLAMA_URLS,
        "ollama_instances": len(_pi.OLLAMA_URLS),
        "any_busy": any(v > 0 for v in activity.values()),
        "pipeline": defs,
        "prompts": {
            "synthesis": CHAT_SYNTHESIS_PROMPT,
            "section_pruning": getattr(_pi, "SECTION_CHECK_PROMPT", ""),
            "leaf_evaluation": getattr(_pi, "LEAF_EVAL_PROMPT", ""),
            "why_not_explainer": getattr(_pi, "EXPLAIN_PROMPT", ""),
        },
    })


class ChatRequest(BaseModel):
    query: str
    index_module: Optional[str] = None
    # Pre-rendered, human-readable background from a client-side structured
    # intake. Kept as free text so the server stays agnostic about which
    # questionnaire produced it.
    context: Optional[str] = None


@app.post("/api/chat")
def chat(req: ChatRequest):
    """The chatbot workflow: the same two-phase retrieval as /api/run, then a
    synthesis call that produces a structured, citation-anchored answer.

    The response carries both the chat payload (answer + sources) and the full
    run payload, so the retrieval tab can render the identical run state.
    """
    query = (req.query or "").strip()
    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)
    patient_context = _context_block(req.context)

    defs = _module_defaults()
    index_mod_name = req.index_module or defs["index"]
    try:
        index_mod = _load_module("index", index_mod_name)
    except Exception as exc:
        return JSONResponse({"error": f"Could not load index module '{index_mod_name}': {exc}"}, status_code=400)

    _set_chat_phase("retrieval", "Reading the document trees…")
    try:
        results = _run_retrieval(query, index_mod)
        eval_errors = _count_eval_errors(results)

        # Nothing retrieved *and* nothing successfully evaluated means retrieval
        # never ran — the model was unreachable. Reporting that as "no passage
        # was judged relevant" states a clinical finding the system never made,
        # and a practitioner could reasonably read it as "no guideline covers
        # this". Fail loudly, before doing any more work on an empty run.
        if not any(doc_data.get("nodes") for doc_data in results.values()) and eval_errors:
            _set_chat_phase("error", "Retrieval could not run")
            return JSONResponse(
                {"error": "The retrieval model is unavailable, so the library "
                          "could not be searched. This is not a finding about "
                          "the documents."},
                status_code=503)

        # Flatten retrieved nodes into numbered sources (stable order: doc, tree order)
        crumbs = _breadcrumbs_for(results)
        sources = []
        for doc_name, doc_data in results.items():
            for node in doc_data["nodes"]:
                pin = node.get("pin") or {}
                sources.append({
                    "id": f"s{len(sources) + 1}",
                    "n": len(sources) + 1,
                    "doc": doc_name,
                    "node_id": node["node_id"],
                    "title": node["title"],
                    "breadcrumb": crumbs.get(doc_name, {}).get(node["node_id"], ""),
                    "excerpt": node["content"],
                    "quote": node["quote"],
                    "reason": node["reason"],
                    "synthetic": node["synthetic"],
                    "pin": node.get("pin"),
                    "page": pin.get("page"),
                    "bbox_normalized": _normalized_bbox(doc_name, pin) if pin else None,
                    "image": pin.get("image"),
                    "has_source_pdf": doc_name in load_sources(),
                })

        answer_text = ""
        sections: dict = {}
        if sources:
            _set_chat_phase("synthesis", f"Composing an answer from {len(sources)} passages…")
            passages = "\n\n".join(
                f"[{s['n']}] {s['doc'].replace('_', ' ')} › {s['breadcrumb'] or s['title']}\n{s['excerpt'] or '(no content)'}"
                for s in sources
            )
            prompt = CHAT_SYNTHESIS_PROMPT.format(
                context_block=patient_context, query=query, passages=passages,
            )
            client = _pi.make_client(_pi.OLLAMA_URLS[0])
            response = client.chat(
                model=getattr(_pi, "SYNTHESIS_MODEL", _pi.MODEL),
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            answer_text = response["message"]["content"]
            # Strip thought blocks
            answer_text = re.sub(r"<thought>.*?(</thought>|$)", "", answer_text, flags=re.S).strip()
            sections = _parse_answer_sections(answer_text)

        cited = set(int(n) for n in re.findall(r"\[(\d+)\]", answer_text))
        valid_cited = {n for n in cited if 1 <= n <= len(sources)}
        if not sources:
            status = "insufficient_evidence"
            summary = "No passage in the library was judged relevant to this question."
        elif valid_cited:
            status = "grounded"
            summary = (f"{len(valid_cited)} of {len(sources)} retrieved passages are "
                       f"cited inline; every source below was selected with a verbatim quote.")
        else:
            status = "partially_grounded"
            summary = (f"{len(sources)} passages were retrieved, but the answer text "
                       f"carries no inline citations — verify against the sources below.")

        if eval_errors:
            # Some of the library was never actually read. Say so rather than
            # letting the summary imply the whole corpus was considered.
            status = "partially_grounded"
            summary = (f"{summary} {eval_errors} passage(s) could not be checked "
                       f"because the model was unavailable, so the library was not "
                       f"fully searched.")

        return JSONResponse({
            "query": query,
            "answer": {
                "content": answer_text,
                "short_answer": sections.get("short_answer", ""),
                "recommended_action": sections.get("recommended_action", ""),
                "rationale": sections.get("rationale", ""),
                "limitations": sections.get("limitations", ""),
            },
            "grounding": {"status": status, "summary": summary, "sources": sources},
            "run": {
                "query": query,
                "results": results,
                "test_result": None,
                "pipeline": {
                    "ingest": defs["ingest"],
                    "index":  index_mod_name,
                    "query":  defs["query"],
                },
            },
        })
    except Exception as exc:
        _set_chat_phase("error", str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        with _chat_lock:
            if _chat_phase["phase"] != "error":
                _chat_phase.update({"phase": "idle", "detail": ""})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_leaves(nodes):
    count = 0
    for n in nodes:
        if n.is_leaf:
            count += 1
        else:
            count += _count_leaves(n.children)
    return count


def _eval_test(test, results):
    expected = test.get("expected", {})
    expected_any = test.get("expected_any", {})

    missing = {
        doc: list(set(ids) - set(results.get(doc, {}).get("retrieved_ids", [])))
        for doc, ids in expected.items()
        if set(ids) - set(results.get(doc, {}).get("retrieved_ids", []))
    }
    any_missing = {
        doc: ids
        for doc, ids in expected_any.items()
        if not (set(ids) & set(results.get(doc, {}).get("retrieved_ids", [])))
    }

    return {
        "passed": not missing and not any_missing,
        "missing": missing,
        "any_missing": any_missing,
        "expected": expected,
        "expected_any": expected_any,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
