"""PageIndex Explorer — FastAPI backend. Serves the UI and wraps pageindex retrieval."""

import atexit
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

# Add project root so we can import pageindex and modules
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pageindex as _pi
from pageindex import _node_from_dict
from modules.registry import discover as _discover_modules, load as _load_module, defaults as _module_defaults

import re

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Ollama instance manager
# ---------------------------------------------------------------------------

BASE_PORT   = 11434
_extra_procs: list[subprocess.Popen] = []   # processes we started on 11435+


def _probe(url: str, timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def _wait_ready(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _probe(url):
            return True
        time.sleep(0.5)
    return False


def _ollama_bin() -> Optional[str]:
    return shutil.which("ollama")


def set_ollama_instances(n: int) -> dict:
    """Start/stop Ollama instances so exactly n are running. Returns status dict."""
    global _extra_procs
    n = max(1, n)

    # Kill any extras we started beyond what's needed
    while len(_extra_procs) > n - 1:
        proc = _extra_procs.pop()
        proc.terminate()
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Start extra instances if needed
    bin_path = _ollama_bin()
    errors = []
    while len(_extra_procs) < n - 1:
        port = BASE_PORT + len(_extra_procs) + 1
        url  = f"http://127.0.0.1:{port}"
        if _probe(url):
            # Already running externally — don't adopt it, just note it
            _extra_procs.append(None)  # placeholder
        elif bin_path:
            env  = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{port}"}
            proc = subprocess.Popen(
                [bin_path, "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _extra_procs.append(proc)
            if not _wait_ready(url, timeout=20):
                errors.append(f"Instance on port {port} did not start in time")
        else:
            errors.append("ollama not found in PATH — cannot start extra instances")
            break

    # Build the URL list and reconfigure pageindex
    urls = [f"http://127.0.0.1:{BASE_PORT + i}" for i in range(n)]
    # Only include instances that are actually responsive
    live_urls = [u for u in urls if _probe(u)]
    _pi.reconfigure_clients(live_urls if live_urls else [f"http://127.0.0.1:{BASE_PORT}"])

    return {
        "requested": n,
        "live": len(live_urls),
        "urls": live_urls,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Dedicated explainer instance — isolated from the retrieval pool so on-demand
# "why not selected" calls never steal a slot from an in-flight query.
# ---------------------------------------------------------------------------

EXPLAINER_PORT = BASE_PORT + 66          # 11500 — reserved for explanations only
_explainer_procs: list[subprocess.Popen] = []
_explainer_lock = threading.Lock()


def ensure_explainer() -> Optional[str]:
    """Return a URL for the explainer instance, lazily starting it on first use.

    Falls back to the base instance only if no ollama binary is available to
    spawn a dedicated one.
    """
    url = f"http://127.0.0.1:{EXPLAINER_PORT}"
    with _explainer_lock:
        if _probe(url):
            return url
        bin_path = _ollama_bin()
        if not bin_path:
            base = f"http://127.0.0.1:{BASE_PORT}"
            return base if _probe(base) else None
        env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{EXPLAINER_PORT}"}
        proc = subprocess.Popen(
            [bin_path, "serve"], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _explainer_procs.append(proc)
        if _wait_ready(url, timeout=25):
            return url
        base = f"http://127.0.0.1:{BASE_PORT}"
        return base if _probe(base) else None


def _shutdown_explainer() -> None:
    for proc in _explainer_procs:
        try:
            proc.terminate()
            proc.wait(timeout=4)
        except Exception:
            try: proc.kill()
            except Exception: pass


atexit.register(_shutdown_explainer)


# Initialise from env on startup
_initial_n = int(os.environ.get("OLLAMA_INSTANCES", "1"))
if _initial_n > 1:
    set_ollama_instances(_initial_n)

app = FastAPI(title="PageIndex Explorer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

# Asset crops extracted by the betteringest_pdf ingest module
# (knowledge_base/assets/<stem>/*.png), referenced from the massaged markdown
# as /assets/<stem>/<file>. check_dir=False: the dir appears on first ingest.
KB_ASSETS_DIR = ROOT / "knowledge_base" / "assets"
app.mount("/assets", StaticFiles(directory=KB_ASSETS_DIR, check_dir=False), name="assets")

SOURCES_MANIFEST = ROOT / "knowledge_base" / ".sources.json"


def _load_sources() -> dict:
    try:
        return json.loads(SOURCES_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}

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
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    html = re.sub(r"/static/[\w.\-]+", lambda m: _versioned(m.group(0)), html)
    return HTMLResponse(html)


@app.get("/api/tests")
def get_tests():
    return JSONResponse(TEST_CASES)


@app.get("/api/status")
def get_status():
    activity = _pi.get_activity()
    return JSONResponse({
        "instances": [
            {"url": url, "index": i, "active": activity.get(url, 0)}
            for i, url in enumerate(_pi.OLLAMA_URLS)
        ],
        "any_busy": any(v > 0 for v in activity.values()),
        "progress": _pi.get_progress(),
        "live":     _pi.get_live_events(),
    })


@app.get("/api/config")
def get_config():
    return JSONResponse({
        "ollama_instances": len(_pi.OLLAMA_URLS),
        "ollama_urls": _pi.OLLAMA_URLS,
        "ollama_bin_available": _ollama_bin() is not None,
    })


class ConfigRequest(BaseModel):
    ollama_instances: int


@app.post("/api/config")
def post_config(req: ConfigRequest):
    result = set_ollama_instances(req.ollama_instances)
    return JSONResponse(result)


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


@app.get("/api/ingest/progress")
def ingest_progress():
    with _ingest_lock:
        return JSONResponse(dict(_ingest_state))


@app.get("/api/document/{stem}/page/{page}")
def get_document_page(stem: str, page: int, bbox: Optional[str] = None,
                      regions: Optional[str] = None):
    """Render one page of the document's SOURCE PDF as a PNG, optionally with
    the pin's bbox highlighted — powers 'jump to its page/bbox' in the UI.

    `bbox` is "x0,y0,x1,y1" and `regions` is JSON [[page,x0,y0,x1,y1],...],
    both in render pixels at the ingest ocr_scale (the pin convention)."""
    src = _load_sources().get(Path(stem).name)
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
    return Response(content=buf.getvalue(), media_type="image/png")


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
