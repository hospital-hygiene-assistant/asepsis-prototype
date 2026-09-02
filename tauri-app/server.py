"""Asepsis Prototype — FastAPI backend. Serves the UI and wraps pageindex retrieval."""

import atexit
import collections
import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Optional

# Add project root so we can import pageindex and modules
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import choices as app_choices
import config as app_config
import debug_cache as app_debug_cache
import tokens as app_tokens
from modules.ingest._manifest import Manifest
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


def _unload_models() -> None:
    """Ask Ollama to drop our models from memory as the app exits.

    Ollama is a separate long-running server: quitting the app (or Ctrl-C in
    the terminal) leaves whatever we loaded resident for the full keep_alive
    window, and a 4-8 GB model sitting in RAM after the app is gone is what
    pushes this machine into swap. A zero keep_alive unloads immediately.

    Best-effort by design: Ollama may already be gone, and failing to tidy up
    must never stop the process exiting.
    """
    models = {getattr(_pi, "MODEL", None), getattr(_pi, "SYNTHESIS_MODEL", None)}
    for url in list(getattr(_pi, "OLLAMA_URLS", [])):
        for model in filter(None, models):
            try:
                req = urllib.request.Request(
                    f"{url.rstrip('/')}/api/generate",
                    data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=3).read()
            except Exception:
                pass


atexit.register(_shutdown_explainer)
atexit.register(_unload_models)


def _on_signal(signum, _frame):
    """atexit does not run on SIGTERM, and a killed app that leaves 4-8 GB of
    model resident is exactly the case worth covering. Ctrl-C (SIGINT) would
    reach atexit on its own; this makes both paths behave the same."""
    _unload_models()
    _shutdown_explainer()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _on_signal)
    except (ValueError, OSError):
        pass          # not the main thread, or unsupported — atexit still covers it


# Initialise from env on startup
def _warm_models() -> None:
    """Load the models into Ollama as the app opens.

    Ollama loads a model on its first request, so without this the opening
    seconds of the FIRST retrieval are spent reading several GB off disk while
    the UI shows a progress bar that is not moving. One throwaway token per
    model gets that out of the way while the user is still reading the page.

    Runs in the background and never raises: a failure here must not stop the
    app from starting, and the first real call would load the model anyway.
    """
    def work():
        models = {getattr(_pi, "MODEL", None), getattr(_pi, "SYNTHESIS_MODEL", None)}
        for url in list(getattr(_pi, "OLLAMA_URLS", []))[:1]:
            for model in filter(None, models):
                try:
                    req = urllib.request.Request(
                        f"{url.rstrip('/')}/api/generate",
                        data=json.dumps({"model": model, "prompt": "ok",
                                         "stream": False, "think": False,
                                         "keep_alive": app_config.keep_alive(),
                                         "options": {"num_predict": 1}}).encode(),
                        headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=300).read()
                    print(f"[warm] {model} resident", file=sys.stderr)
                except Exception as exc:
                    print(f"[warm] could not preload {model}: {exc}", file=sys.stderr)

    threading.Thread(target=work, daemon=True).start()


_warm_models()

_initial_n = int(os.environ.get("OLLAMA_INSTANCES", "1"))
if _initial_n > 1:
    set_ollama_instances(_initial_n)

app = FastAPI(title="Asepsis Prototype")
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
        "pool":     {"workers": _pi.pool_size()},
        "tokens":   app_tokens.calibration(),
        "index":    _index_state(),
    })


# ---------------------------------------------------------------------------
# Index format migration
#
# v1 indexes carry no summaries and no content hashes, so they cannot be
# upgraded in place — retrieval quality now depends on summaries that simply
# are not there. A stale corpus blocks retrieval until it is rebuilt; the cost
# is paid exactly once, and every rebuild after that is incremental.
# ---------------------------------------------------------------------------

_reindex_lock = threading.Lock()
_reindex_state: dict = {"state": "idle", "done": 0, "total": 0, "doc": "", "message": ""}


def _index_dir() -> Path:
    index_mod = _load_module("index", _module_defaults()["index"])
    return getattr(index_mod, "INDEX_DIR", Path("index"))


def _index_state() -> dict:
    stale = _pi.stale_indexes(_index_dir())
    with _reindex_lock:
        rebuild = dict(_reindex_state)
    return {
        "format_version": app_config.INDEX_FORMAT_VERSION,
        "stale_docs": stale,
        "stale": bool(stale),
        "rebuild": rebuild,
    }


@app.get("/api/index/state")
def index_state():
    return JSONResponse(_index_state())


@app.post("/api/index/rebuild")
def rebuild_index():
    """Rebuild every document's index in the current format."""
    with _reindex_lock:
        if _reindex_state["state"] == "running":
            return JSONResponse({"error": "a rebuild is already in progress"}, status_code=409)
        _reindex_state.update({"state": "running", "done": 0, "total": 0,
                               "doc": "", "message": "Starting…"})

    index_mod = _load_module("index", _module_defaults()["index"])
    ingest_mod = _load_module("ingest", _module_defaults()["ingest"])
    kb_dir = getattr(ingest_mod, "KB_DIR", Path("knowledge_base"))
    if not Path(kb_dir).is_absolute():
        kb_dir = ROOT / kb_dir
    stems = sorted(p.stem for p in Path(kb_dir).glob("*.md"))

    def work():
        try:
            with _reindex_lock:
                _reindex_state["total"] = len(stems)
            for i, stem in enumerate(stems):
                with _reindex_lock:
                    _reindex_state.update({
                        "doc": stem, "done": i,
                        "message": f"Summarising {stem} ({i + 1}/{len(stems)})…"})
                index_mod.build_index(stem)
            with _reindex_lock:
                _reindex_state.update({"state": "done", "done": len(stems),
                                       "message": "Index rebuilt"})
        except Exception as exc:
            with _reindex_lock:
                _reindex_state.update({"state": "error", "message": str(exc)})

    threading.Thread(target=work, daemon=True).start()
    return JSONResponse({"started": True, "docs": stems})


def _stale_index_response():
    """409 body shared by the retrieval endpoints when the corpus is stale."""
    stale = _pi.stale_indexes(_index_dir())
    if not stale:
        return None
    return JSONResponse({
        "error": "index_stale",
        "message": ("The index predates the current format and has no summaries. "
                    "Rebuild it before querying."),
        "stale_docs": stale,
    }, status_code=409)


@app.get("/api/config")
def get_config():
    return JSONResponse({
        "ollama_instances": len(_pi.OLLAMA_URLS),
        "ollama_urls": _pi.OLLAMA_URLS,
        "ollama_bin_available": _ollama_bin() is not None,
        **app_config.describe(),
    })


class ConfigRequest(BaseModel):
    ollama_instances: Optional[int] = None
    retrieval_model: Optional[str] = None
    synthesis_model: Optional[str] = None
    # Context windows, in tokens. The agent window is the one that decides how
    # many retrieved passages survive to the answer (see the Phase-3 budget),
    # so it is deliberately user-adjustable at runtime.
    retrieval_ctx: Optional[int] = None
    agent_ctx: Optional[int] = None
    concurrency_per_instance: Optional[int] = None
    max_leaf_evals: Optional[int] = None
    debug_cache_enabled: Optional[bool] = None


@app.post("/api/config")
def post_config(req: ConfigRequest):
    result = (set_ollama_instances(req.ollama_instances)
              if req.ollama_instances is not None else {})

    prev_concurrency = app_config.runtime().concurrency_per_instance
    app_config.update(
        retrieval_model=req.retrieval_model,
        synthesis_model=req.synthesis_model,
        retrieval_ctx=req.retrieval_ctx,
        agent_ctx=req.agent_ctx,
        concurrency_per_instance=req.concurrency_per_instance,
        max_leaf_evals=req.max_leaf_evals,
        debug_cache_enabled=req.debug_cache_enabled,
    )
    if req.retrieval_model:
        _pi.MODEL = req.retrieval_model
    if req.synthesis_model:
        _pi.SYNTHESIS_MODEL = req.synthesis_model
    if (req.concurrency_per_instance is not None
            and req.concurrency_per_instance != prev_concurrency):
        _pi._reset_work_pool()

    result.update(app_config.describe())
    result["ollama_instances"] = len(_pi.OLLAMA_URLS)
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


def _manifest() -> Manifest:
    return Manifest(ROOT / "knowledge_base")


# ---------------------------------------------------------------------------
# Multiple-choice pre-filter
#
# The questions shown before the user types. Each answer maps to a set of
# leaves judged relevant to it once, at index time; the selections are unioned
# at query time. Precompute runs on its own RunContext so it can never disturb
# the live visualisation of a query in flight.
# ---------------------------------------------------------------------------

_facet_lock = threading.Lock()
_facet_state: dict = {"state": "idle", "doc": "", "done": 0, "total": 0,
                      "calls": 0, "reused": 0, "message": "", "errors": []}
_facet_cancel = threading.Event()


def _leaves_of(stem: str):
    """A document's leaves, or None when its index cannot be read."""
    try:
        return _pi._collect_leaves(_pi.load_index_nodes(stem))
    except (OSError, FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def _facet_coverage() -> dict:
    """What the judgements on DISK actually cover.

    The job's own progress is a process variable, so it said "not computed
    yet" after every restart and never noticed a document ingested since the
    last run. Disk is the only thing that knows, so the UI is driven from
    here and the in-flight job is layered on top.
    """
    index_dir = _index_dir()
    questions = app_choices.load_questions()
    docs = []
    for path in sorted(index_dir.glob("*.json")):
        leaves = _leaves_of(path.stem)
        if leaves is None:
            continue
        docs.append(app_choices.document_coverage(
            index_dir, path.stem, leaves, _pi.MODEL, questions))

    by_state = collections.Counter(d["state"] for d in docs)
    return {
        "docs": docs,
        "ready": by_state["ready"],
        "needs_work": len(docs) - by_state["ready"],
        "leaves": sum(d["leaves"] for d in docs),
        "judged": sum(d["judged"] for d in docs),
        # The single question the UI actually asks: can these answers filter?
        "usable": bool(docs) and by_state["ready"] > 0,
        "complete": bool(docs) and by_state["ready"] == len(docs),
    }


def _facet_status() -> dict:
    with _facet_lock:
        job = dict(_facet_state)
    return {**job, "coverage": _facet_coverage()}


@app.get("/api/choices")
def get_choices():
    qs = app_choices.load_questions()
    return JSONResponse({**qs.to_dict(),
                         "default_mode": app_choices.ANY,
                         "modes": list(app_choices.MODES),
                         "precompute": _facet_status()})


@app.get("/api/choices/precompute")
def facet_precompute_status():
    return JSONResponse(_facet_status())


@app.get("/api/choices/candidates")
def choice_candidates(selected_answers: str = "", mode: str = app_choices.ANY,
                      tags: str = ""):
    """How many passages the selected answers put in scope.

    Honours the tag filter, because the count has to describe the corpus the
    query will ACTUALLY run over — counting documents the tags exclude made
    the number quietly wrong whenever both filters were in use.
    """
    picked = [a for a in (selected_answers or "").split(",") if a]
    tag_list = [t for t in (tags or "").split(",") if t]
    mode = app_choices.normalise_mode(mode)
    questions = app_choices.load_questions()
    index_dir = _index_dir()
    allowed = _manifest().filter_docs(tag_list) if tag_list else None

    total = candidates = unjudged = 0
    for path in sorted(index_dir.glob("*.json")):
        if allowed is not None and path.stem not in allowed:
            continue
        leaves = _leaves_of(path.stem)
        if leaves is None:
            continue
        total += len(leaves)
        if not picked:
            candidates += len(leaves)
            continue
        store = app_choices.open_store(index_dir, path.stem, _pi.MODEL, questions)
        selection = app_choices.select_leaves(store, leaves, picked, questions, mode)
        candidates += len(selection.kept)
        unjudged += len(selection.unjudged)
    return JSONResponse({"candidates": candidates, "total": total,
                         "unjudged": unjudged, "mode": mode,
                         "selected_answers": picked})


@app.post("/api/choices/precompute")
def facet_precompute():
    """Judge every leaf against every question. Resumable and incremental:
    unchanged leaves reuse their stored judgements."""
    with _facet_lock:
        if _facet_state["state"] == "running":
            return JSONResponse({"error": "a precompute is already running"}, status_code=409)
        _facet_cancel.clear()
        _facet_state.update({"state": "running", "doc": "", "done": 0, "total": 0,
                             "calls": 0, "reused": 0, "message": "Starting…",
                             "errors": []})

    index_dir = _index_dir()
    stems = sorted(p.stem for p in index_dir.glob("*.json"))

    def work():
        # Its own context: a background job must not clobber a live query's
        # progress or treemap.
        ctx = _pi.new_run(make_current=False)
        # Judgements are independent, so they go on the work pool — the same
        # bounded pool retrieval uses, so a precompute cannot outrun the
        # Ollama instances. THIS thread only coordinates and writes the store.
        pool = _pi.work_pool()
        try:
            for i, stem in enumerate(stems):
                if _facet_cancel.is_set():
                    break
                with _facet_lock:
                    _facet_state.update({"doc": stem, "done": i, "total": len(stems),
                                         "message": f"Judging {stem} ({i + 1}/{len(stems)})…"})
                leaves = _leaves_of(stem)
                if leaves is None:
                    with _facet_lock:
                        _facet_state["errors"].append(f"{stem}: index unreadable")
                    continue
                def on_progress(p, stem=stem, i=i):
                    # Judgement-level, not document-level: a long document used
                    # to leave the notice frozen on its name for minutes.
                    where = f"{stem} ({i + 1}/{len(stems)})"
                    if p["total"]:
                        note = f"Judging {where} — {p['done']}/{p['total']} judgements"
                    else:
                        note = f"{where} already judged"
                    with _facet_lock:
                        _facet_state["message"] = note

                report = app_choices.precompute_document(
                    stem, leaves, index_dir,
                    lambda prompt: _pi._chat(prompt, kind="facet"),
                    _pi._parse_json_response, _pi.MODEL,
                    progress=on_progress,
                    submit=pool.submit,
                    # Enough to keep every instance busy, short enough that a
                    # question asked mid-precompute is not queued behind the
                    # whole corpus.
                    max_inflight=_pi.pool_size() * 2,
                    cancelled=_facet_cancel.is_set)
                with _facet_lock:
                    _facet_state["calls"] += report["calls"]
                    _facet_state["reused"] += report["reused"]
                    _facet_state["errors"].extend(report["errors"][:20])
            with _facet_lock:
                cancelled = _facet_cancel.is_set()
                _facet_state.update({
                    "state": "cancelled" if cancelled else "done",
                    "done": len(stems),
                    "message": ("Stopped — the judgements already made are kept"
                                if cancelled else "Pre-filter judgements ready")})
        except Exception as exc:
            with _facet_lock:
                _facet_state.update({"state": "error", "message": str(exc)})
        finally:
            del ctx

    threading.Thread(target=work, daemon=True).start()
    return JSONResponse({"started": True, "docs": stems})


@app.post("/api/choices/precompute/cancel")
def facet_precompute_cancel():
    """Stop the pass. Everything judged so far stays on disk — the store is
    incremental, so a cancelled run is progress, not waste."""
    _facet_cancel.set()
    with _facet_lock:
        running = _facet_state["state"] == "running"
        if running:
            _facet_state["message"] = "Stopping…"
    return JSONResponse({"cancelling": running})


# ---------------------------------------------------------------------------
# Debug cache
#
# Exact-key replay of a previous run, for iterating on prompts without paying
# for retrieval each time. Not a production cache: no similarity matching.
# ---------------------------------------------------------------------------

_cache = app_debug_cache.DebugCache()


def _cache_key_for(query: str, tags=None, selected_answers=None,
                   selection_mode: str = app_choices.ANY) -> str:
    cfg = app_config.runtime()
    # The mode is part of the key: the same answers in "all" retrieve a
    # different set than in "any", so replaying one as the other would serve
    # a narrower answer than the user asked for.
    answers = list(selected_answers or [])
    if answers:
        answers = answers + [f"__mode:{app_choices.normalise_mode(selection_mode)}"]
    return app_debug_cache.make_key(
        query, _index_dir(), tags=tags, selected_answers=answers,
        retrieval_model=cfg.retrieval_model, synthesis_model=cfg.synthesis_model)


@app.get("/api/cache/lookup")
def cache_lookup(query: str, tags: Optional[str] = None,
                 selected_answers: Optional[str] = None,
                 selection_mode: str = app_choices.ANY):
    """Has this exact question been run against this exact corpus before?

    The frontend calls this on submit and offers "replay or re-run live".
    """
    if not _cache.enabled:
        return JSONResponse({"hit": False, "enabled": False})
    tag_list = [t for t in (tags or "").split(",") if t]
    answer_list = [a for a in (selected_answers or "").split(",") if a]
    key = _cache_key_for(query, tag_list, answer_list, selection_mode)
    entry = _cache.get(key)
    return JSONResponse({"hit": entry is not None, "enabled": True, "key": key,
                         **(entry.summary() if entry else {})})


@app.get("/api/cache/questions")
def cache_questions():
    """Previously-asked questions that are still replayable against the
    current index, for the command bar's autocomplete."""
    return JSONResponse({
        "enabled": _cache.enabled,
        "questions": [e.summary() for e in _cache.entries(_index_dir())],
    })


@app.post("/api/cache/clear")
def cache_clear():
    return JSONResponse({"cleared": _cache.clear()})


@app.get("/api/tags")
def get_tags():
    """Every tag in the library, with how many documents carry it.

    Tags come from the folder each document was ingested from, so this is the
    corpus's own notion of provenance (internal vs arxiv vs guidelines).
    """
    return JSONResponse({"tags": _manifest().all_tags()})


class TagUpdateRequest(BaseModel):
    doc_id: str
    manual_tags: list[str]


@app.post("/api/tags")
def set_tags(req: TagUpdateRequest):
    """Set a document's curated tags. Folder-derived tags are untouched — a
    re-ingest refreshes those and must never clobber hand curation."""
    manifest = _manifest()
    entry = manifest.set_manual_tags(req.doc_id, req.manual_tags)
    if entry is None:
        return JSONResponse({"error": f"unknown document '{req.doc_id}'"}, status_code=404)
    return JSONResponse({"doc_id": entry.doc_id, "tags": entry.effective_tags,
                         "manual_tags": entry.manual_tags})


def _folder_of(entry) -> list[str]:
    """The folder path a document was ingested from, outermost first.

    Empty for a document at the top level of the source folder — it belongs to
    the library itself, not to a folder inside it.
    """
    if entry is None or not entry.rel_path:
        return []
    parts = PurePosixPath(entry.rel_path.replace("\\", "/")).parent.parts
    return [p for p in parts if p not in (".", "/", "")]


@app.get("/api/documents")
def get_documents():
    # Use the default index module to find where indices live
    index_mod = _load_module("index", _module_defaults()["index"])
    index_dir = getattr(index_mod, "INDEX_DIR", Path("index"))
    manifest = _manifest()
    docs = []
    for idx in sorted(index_dir.glob("*.json")):
        # The frontend consumes a bare node array; the on-disk format wraps it
        # with build provenance (see pageindex.read_index_file).
        tree = _pi.read_index_file(idx)["nodes"]
        nodes = [_node_from_dict(d) for d in tree]
        entry = manifest.get(idx.stem)
        docs.append({
            "name": idx.stem,
            "doc_id": idx.stem,
            "title": entry.title if entry else idx.stem,
            "tags": entry.effective_tags if entry else [],
            # The ORDERED folder path this document was ingested from, which
            # `tags` cannot express: effective_tags is a sorted set unioned
            # with manual tags, so "papers/arxiv" and a hand-typed "arxiv"
            # are indistinguishable in it. The treemap groups on this.
            "folder": _folder_of(entry),
            "rel_path": entry.rel_path if entry else "",
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


_active_reviews = {}


class ConfirmAsset(BaseModel):
    type: str  # "figure" | "table"
    page: int  # 0-based page index
    bbox: list[float]  # [x0, y0, x1, y1]
    name: Optional[str] = None


class ConfirmDoc(BaseModel):
    stem: str
    assets: list[ConfirmAsset]
    non_assets: Optional[list[dict]] = None


class ConfirmRequest(BaseModel):
    docs: list[ConfirmDoc]


@app.post("/api/ingest/cancel")
def cancel_ingest():
    """Cancel the active layout review suspension and reset the state."""
    with _ingest_lock:
        _ingest_state.update({
            "state": "error",
            "phase": "cancelled",
            "message": "Ingestion cancelled by user.",
            "warnings": [],
            "docs": []
        })
        _active_reviews.clear()
    return {"ok": True}


@app.post("/api/ingest/confirm")
def confirm_ingest(req: ConfirmRequest):
    """Resume and finalize ingestion with the user-confirmed layout assets."""
    with _ingest_lock:
        if _ingest_state["state"] != "awaiting_review":
            return JSONResponse({"error": "No ingest review is pending"}, status_code=400)
        _ingest_state.update({"state": "running", "phase": "finalizing", "message": "Finalizing layout adjustments…"})

    def work():
        try:
            import modules.ingest.betteringest_pdf as bpdf
            
            docs_done = []
            for doc_req in req.docs:
                stem = doc_req.stem
                if stem not in _active_reviews:
                    continue
                review_data = _active_reviews[stem]
                
                confirmed_assets_dicts = [
                    {"type": a.type, "page": a.page, "bbox": a.bbox, "name": a.name}
                    for a in doc_req.assets
                ]
                
                non_assets_dicts = doc_req.non_assets if doc_req.non_assets is not None else review_data["non_assets"]
                
                res = bpdf.finalize_ingestion(
                    pdf_path=review_data["pdf_path"],
                    confirmed_assets=confirmed_assets_dicts,
                    non_assets=non_assets_dicts,
                    progress_cb=lambda msg: _set_ingest_state(phase="captioning", message=msg)
                )
                docs_done.append(res["stem"])
                
            _set_ingest_state(phase="index", message="Rebuilding index…")
            index_name = _module_defaults()["index"]
            index_mod = _load_module("index", index_name)
            kb_dir = Path("knowledge_base")
            if not kb_dir.is_absolute():
                kb_dir = ROOT / kb_dir
            for md in sorted(kb_dir.glob("*.md")):
                index_mod.build_index(md.stem)
                
            _active_reviews.clear()
            _set_ingest_state(state="done", phase="done", message="Ingest complete with custom layout modifications", docs=docs_done)
        except Exception as exc:
            _set_ingest_state(state="error", message=str(exc))

    threading.Thread(target=work, daemon=True).start()
    return JSONResponse({"started": True})


@app.get("/api/render_pdf_page")
def render_pdf_page(path: str, page: int, scale: float = 2.0):
    """Render a specific page of an arbitrary local PDF file as a PNG (used by layout review canvas)."""
    import io
    pdf_path = Path(path).expanduser()
    if not pdf_path.exists():
        return JSONResponse({"error": f"PDF file not found: {pdf_path}"}, status_code=404)
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        return JSONResponse({"error": f"Needs pypdfium2: {exc}"}, status_code=501)
    try:
        doc = pdfium.PdfDocument(str(pdf_path))
        if not (1 <= page <= len(doc)):
            return JSONResponse({"error": f"page {page} out of range"}, status_code=404)
        img = doc[page - 1].render(scale=scale).to_pil().convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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
            
            if ingest_name == "betteringest_pdf":
                import modules.ingest.betteringest_pdf as bpdf
                from modules.ingest._manifest import discover_files

                _set_ingest_state(phase="ocr", message="Performing layout detection...")
                src_path = source_dir or getattr(ingest_mod, "get_source_dir")()
                if not src_path:
                    raise FileNotFoundError("BetterIngest source folder not set.")
                # Recursive, like every other ingest path — sub-folder names are
                # what become tags, and a flat glob silently ignored them.
                pdfs = discover_files(Path(src_path), (".pdf",))
                if not pdfs:
                    raise FileNotFoundError(f"No PDF files found under {src_path}/")
                
                review_results = bpdf.prepare_layout_review(
                    [str(p) for p in pdfs],
                    progress_cb=lambda page, total, stem: _set_ingest_state(
                        phase="ocr",
                        message=f"Layout detecting {stem}: page {page + 1}/{total}."
                    )
                )
                
                with _ingest_lock:
                    _active_reviews.clear()
                    for r in review_results:
                        _active_reviews[r["stem"]] = r
                    
                    review_docs = [
                        {
                            "stem": r["stem"],
                            "pdf_path": r["pdf_path"],
                            "pages": r["pages"],
                            "detected_assets": r["detected_assets"],
                            "non_assets": r["non_assets"]
                        }
                        for r in review_results
                    ]
                    _ingest_state.update({
                        "state": "awaiting_review",
                        "phase": "review",
                        "message": "Awaiting layout confirmation from user...",
                        "review_docs": review_docs
                    })
                return

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
    library on top of the existing corpus, then re-index."""
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
            if req.ingest_module == "betteringest_pdf":
                _set_ingest_state(phase="ocr", message="Performing layout detection...")
                import modules.ingest.betteringest_pdf as bpdf
                review_results = bpdf.prepare_layout_review(
                    [str(p) for p in pdfs],
                    progress_cb=lambda page, total, stem: _set_ingest_state(
                        phase="ocr",
                        message=f"Layout detecting {stem}: page {page + 1}/{total}."
                    )
                )
                
                with _ingest_lock:
                    _active_reviews.clear()
                    for r in review_results:
                        _active_reviews[r["stem"]] = r
                    
                    review_docs = [
                        {
                            "stem": r["stem"],
                            "pdf_path": r["pdf_path"],
                            "pages": r["pages"],
                            "detected_assets": r["detected_assets"],
                            "non_assets": r["non_assets"]
                        }
                        for r in review_results
                    ]
                    _ingest_state.update({
                        "state": "awaiting_review",
                        "phase": "review",
                        "message": "Awaiting layout confirmation from user...",
                        "review_docs": review_docs
                    })
                return

            result = ingest_mod.run_paths(
                [str(x) for x in pdfs],
                progress=lambda info: _set_ingest_state(**info)) or {}

            _set_ingest_state(phase="index", message="Rebuilding index…")
            index_mod = _load_module("index", index_name)
            kb_dir = getattr(ingest_mod, "KB_DIR", Path("knowledge_base"))
            if not Path(kb_dir).is_absolute():
                kb_dir = ROOT / kb_dir
            for stem in result.get("docs", []):
                index_mod.build_index(stem)

            _set_ingest_state(state="done", phase="done",
                              message="Additive Ingest complete",
                              warnings=result.get("warnings", []),
                              docs=result.get("docs", []))
        except Exception as exc:
            _set_ingest_state(state="error", message=str(exc))

    threading.Thread(target=work, daemon=True).start()
    return JSONResponse({"started": True, "pdf_count": len(pdfs), "pdfs": [x.name for x in pdfs]})


@app.get("/api/ingest/progress")
def ingest_progress():
    with _ingest_lock:
        return JSONResponse(dict(_ingest_state))


# Rendered-page cache — chat citation previews request the same page/bbox
# repeatedly; pdfium renders are ~100ms each, so memoize the PNG bytes.
_page_png_cache: dict = {}
_page_png_lock = threading.Lock()
_PAGE_CACHE_MAX = 64


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

    nodes = [_node_from_dict(d) for d in _pi.read_index_file(idx)["nodes"]]
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
    # Restrict retrieval to documents carrying ANY of these tags. Empty = all.
    tags: Optional[list[str]] = None
    # Answer ids from the multiple-choice pre-filter.
    selected_answers: Optional[list[str]] = None
    # How those answers combine: "any" (default, recall-first — a passage
    # matching ANY answer is searched) or "all" (Fast mode — a passage must
    # satisfy every answered question). Never inferred; always the user's.
    selection_mode: str = "any"
    # Debug cache: replay a previous identical run instead of re-retrieving.
    use_cache: bool = False
    # Module selections — default to pipeline defaults when not supplied
    ingest_module: Optional[str] = None
    index_module:  Optional[str] = None
    query_module:  Optional[str] = None


def _replay(entry) -> tuple[dict, object]:
    """Rebuild run state from a cache entry.

    The live-event snapshot has to be restored, not just the results: starting
    a run clears every event set, so without this a cache hit would render an
    empty treemap and an empty reasoning trace.
    """
    total = sum(len(d.get("retrieved_ids") or []) for d in entry.results.values())
    ctx = _pi.start_run(total)
    ctx.restore(entry.events)
    return entry.results, ctx


def _budget_payload(ctx, results: dict) -> dict:
    """What the agent's context could and could not hold.

    `deferred` nodes were never evaluated — the ranking ran past the budget
    before reaching them. That is materially different from being pruned or
    rejected, so they are reported separately and stay browsable.
    """
    deferred = []
    for doc_name, doc_data in results.items():
        for node_id, meta in (doc_data.get("node_meta") or {}).items():
            if meta.get("status") != "deferred":
                continue
            node = next((n for n in _iter_tree(doc_data["tree"]) if n["nodeId"] == node_id), None)
            deferred.append({
                "doc": doc_name,
                "node_id": node_id,
                "title": (node or {}).get("title", node_id),
                "reason": meta.get("reason", ""),
                "bm25_score": meta.get("bm25_score"),
                "bm25_rank": meta.get("bm25_rank"),
            })
    deferred.sort(key=lambda d: (d["bm25_rank"] is None, d["bm25_rank"]))

    b = ctx.budget
    return {
        "evaluated": b.evaluated,
        "deferred": len(deferred),
        "tokens_used": b.tokens_used,
        "tokens_max": b.tokens_max,
        "capped_by": b.capped_by,
        "agent_ctx": app_config.runtime().resolved_agent_ctx(),
        "deferred_nodes": deferred,
    }


def _iter_tree(nodes):
    for n in nodes:
        yield n
        yield from _iter_tree(n.get("children", []))


def _run_retrieval(query: str, index_mod, ctx=None, tags=None,
                   selected_answers=None,
                   selection_mode: str = app_choices.ANY) -> tuple[dict, object]:
    """Two-phase retrieval across every indexed document. Shared by /api/run
    (retrieval tab) and /api/chat (chatbot tab) so both drive the same live
    treemap/progress state.

    Documents are retrieved in parallel through pageindex's shared bounded
    pool — previously this loop was serial, so retrieval time grew linearly
    with corpus size even when the Ollama instances were idle.
    """
    index_dir = getattr(index_mod, "INDEX_DIR", Path("index"))

    # Parse each index exactly once: the leaf count (progress denominator) and
    # the raw tree returned to the frontend both come from this one read.
    # Tag filtering happens HERE, before anything else: skipping a document
    # costs nothing, whereas every document that survives this line will cost
    # LLM calls. It is the cheapest defence against corpus growth there is.
    allowed = _manifest().filter_docs(tags) if tags else None

    trees: dict[str, list] = {}
    for idx in sorted(index_dir.glob("*.json")):
        if allowed is not None and idx.stem not in allowed:
            continue
        try:
            trees[idx.stem] = _pi.read_index_file(idx)["nodes"]
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[retrieval] skipping unreadable index {idx.name}: {exc}", file=sys.stderr)

    total_leaves = sum(
        _count_leaves([_node_from_dict(d) for d in tree]) for tree in trees.values()
    )
    ctx = ctx or _pi.start_run(total_leaves)
    ctx.set_total(total_leaves)

    # Pruning runs per document, in parallel. Leaf evaluation does NOT: every
    # retrieved passage lands in one agent context, so candidates are ranked
    # and budgeted across the whole corpus rather than per file.
    supports_two_phase = (hasattr(index_mod, "prune_document")
                          and hasattr(index_mod, "evaluate_ranked"))
    # Per-document orchestration goes on the COORDINATION pool. These tasks
    # block on the node-level calls they submit, so running them on the work
    # pool deadlocks it (see pageindex.coordination_pool).
    pool = _pi.coordination_pool()

    per_doc: dict[str, tuple] = {}
    choice_by_doc: dict[str, dict] = {}
    if supports_two_phase:
        prune = index_mod.prune_document
        # A third-party index module may predate the pre-filter entirely.
        takes_mode = "selection_mode" in inspect.signature(prune).parameters
        futures = {
            pool.submit(prune, doc_name, query, ctx, selected_answers,
                        **({"selection_mode": selection_mode} if takes_mode else {})):
                doc_name
            for doc_name in trees}
        candidates = []
        for future, doc_name in futures.items():
            try:
                doc = future.result()
            except FileNotFoundError:
                continue
            except Exception as exc:
                print(f"[retrieval] {doc_name} failed: {exc}", file=sys.stderr)
                continue
            candidates.append(doc)

        # The one genuinely suspicious case: EVERY document pruned away. A
        # single document doing so is a normal, useful verdict; all of them
        # doing so is either a real "nothing here" or a bad model day, and the
        # difference matters enough to say out loud rather than silently
        # returning an empty answer.
        if candidates and not any(d.candidates for d in candidates):
            print(f"[retrieval] every document pruned to nothing for "
                  f"{query!r} — no passage was judged worth reading",
                  file=sys.stderr)

        index_mod.evaluate_ranked(candidates, query, ctx)

        for doc in candidates:
            selected = [l for l in doc.leaves
                        if doc.node_meta.get(l.node_id, {}).get("relevant")]
            per_doc[doc.doc_name] = (selected, doc.node_meta)
            if getattr(doc, "choice_filter", None):
                choice_by_doc[doc.doc_name] = {
                    "doc": doc.doc_name,
                    **doc.choice_filter,
                    "tags": getattr(doc, "choice_tags", {}) or {},
                }
    else:
        # A third-party index module without the two-phase API.
        retrieve_fn = getattr(index_mod, "retrieve_with_metadata", None)
        futures = {pool.submit(
            retrieve_fn or (lambda d, q: (index_mod.retrieve(d, q), {})),
            doc_name, query): doc_name for doc_name in trees}
        for future, doc_name in futures.items():
            try:
                per_doc[doc_name] = future.result()
            except FileNotFoundError:
                continue
            except Exception as exc:
                print(f"[retrieval] {doc_name} failed: {exc}", file=sys.stderr)

    results = {}
    for doc_name in trees:
        if doc_name not in per_doc:
            continue
        nodes, node_reasons = per_doc[doc_name]
        raw_tree = trees[doc_name]
        results[doc_name] = {
            "tree": raw_tree,
            "retrieved_ids": [n.node_id for n in nodes],
            "node_meta": node_reasons,  # {node_id: {reason, quote}}
            # What the multiple-choice pre-filter did here. Absent when the
            # user selected nothing.
            "choice_filter": choice_by_doc.get(doc_name, {}),
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
    return results, ctx


def _choice_payload(results: dict, selected_answers=None,
                    selection_mode: str = app_choices.ANY) -> dict:
    """Corpus-level account of what the pre-filter did.

    The per-document verdicts are already in `results`; this is the one-line
    version the UI shows above an answer. `unfiltered` is the entry that
    earns its place: a document searched whole for want of judgements looks
    identical to a filtered one unless it is counted here.
    """
    per_doc = [d.get("choice_filter") or {} for d in results.values()]
    applied = [c for c in per_doc if c.get("applied")]
    if not applied:
        return {"applied": False, "mode": selection_mode,
                "selected_answers": selected_answers or []}
    return {
        "applied": True,
        "mode": selection_mode,
        "selected_answers": selected_answers or [],
        "kept": sum(c.get("kept", 0) for c in applied),
        "total": sum(c.get("total", 0) for c in applied),
        "docs_excluded": sum(1 for c in applied if c.get("state") == "excluded"),
        "docs_unfiltered": [c["doc"] for c in applied
                            if c.get("state") == "unfiltered"],
    }


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

    stale = _stale_index_response()
    if stale is not None:
        return stale

    # Resolve modules
    defs = _module_defaults()
    index_mod_name = req.index_module or defs["index"]
    try:
        index_mod = _load_module("index", index_mod_name)
    except Exception as exc:
        return JSONResponse({"error": f"Could not load index module '{index_mod_name}': {exc}"}, status_code=400)

    cache_key = _cache_key_for(query, req.tags, req.selected_answers,
                               req.selection_mode)
    cached = _cache.get(cache_key) if req.use_cache else None
    if cached is not None:
        results, ctx = _replay(cached)
        cache_info = {"cached": True, **cached.summary()}
    else:
        results, ctx = _run_retrieval(query, index_mod, tags=req.tags,
                                      selected_answers=req.selected_answers,
                                      selection_mode=req.selection_mode)
        _cache.put(cache_key, query=query, results=results,
                   events=ctx.events_snapshot(),
                   budget=_budget_payload(ctx, results),
                   tags=req.tags, selected_answers=req.selected_answers,
                   index_dir=_index_dir())
        cache_info = {"cached": False, "key": cache_key}

    test_result = _eval_test(test, results) if test else None
    return JSONResponse({
        "query": query,
        "results": results,
        "budget": _budget_payload(ctx, results),
        "choices": _choice_payload(results, req.selected_answers,
                                   req.selection_mode),
        "cache": cache_info,
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

# Set by /api/chat/cancel. Retrieval checks it between node calls and the
# synthesis stream breaks on it — breaking closes the HTTP connection, which
# is what makes Ollama stop generating rather than finishing into the void.
_chat_cancel = threading.Event()


class ChatCancelled(Exception):
    """Raised to unwind a run the user stopped."""


_chat_lock = threading.Lock()
_chat_phase: dict = {"phase": "idle", "detail": ""}
# The retrieved passages of the run in flight, for the evidence deck.
_chat_cards: list = []


def _set_chat_phase(phase: str, detail: str = "", **extra) -> None:
    with _chat_lock:
        # `extra` carries the retrieved passages once retrieval is done, so the
        # UI can show them while synthesis runs. Reset on every transition or
        # a stale phase would keep serving the previous question's cards.
        _chat_phase.clear()
        _chat_phase.update({"phase": phase, "detail": detail, **extra})


CHAT_SYNTHESIS_PROMPT = """\
You are a clinical knowledge assistant. Answer the question using ONLY the
numbered source passages below. Every claim must cite its passage inline as
[n] (e.g. [1] or [2][3]). Never invent a source number.

Question: {query}

Source passages:
{passages}

Respond in EXACTLY this format (keep the labels, fill in the text; each
section is 1-3 sentences of plain text with inline [n] citations):

SHORT_ANSWER: <the direct answer to the question>
RECOMMENDED_ACTION: <what the clinician should do>
RATIONALE: <why, grounded in the cited passages>
LIMITATIONS: <what the sources do not cover, or uncertainty>

If the passages cannot answer the question, say so in SHORT_ANSWER and leave
the other sections brief.
"""

_SECTION_KEYS = ["SHORT_ANSWER", "RECOMMENDED_ACTION", "RATIONALE", "LIMITATIONS"]


def _parse_answer_sections(text: str) -> dict:
    """Parse the labeled sections out of the model's answer. Tolerant of
    markdown bolding and missing sections; anything unmatched stays in
    'content' as the fallback body."""
    pattern_keys = [k.replace("_", r"[\s_-]?") for k in _SECTION_KEYS]
    pattern = re.compile(
        r"^\s*(?:#+\s*)?\**\s*(" + "|".join(pattern_keys) + r")\s*\**\s*:?\s*",
        re.MULTILINE | re.IGNORECASE,
    )
    sections: dict[str, str] = {}
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        raw_key = m.group(1)
        key = raw_key.upper().replace(" ", "_").replace("-", "_")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[key.lower()] = text[m.end():end].strip().strip("*").strip()
    return sections


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


@app.post("/api/chat/cancel")
def chat_cancel():
    """Stop the run in flight.

    Retrieval checks the flag between node calls and synthesis breaks out of
    its stream, which closes the connection — so this actually stops the model
    working rather than just hiding the result from the user.
    """
    _chat_cancel.set()
    _pi.request_cancel()
    return JSONResponse({"cancelling": True})


@app.get("/api/chat/cards")
def chat_cards():
    """The passages retrieved by the run in flight.

    Fetched once when the status poll reports them, so the poll itself stays
    small: these carry the full excerpt and page pin, which is what lets a
    card show the same highlight and open the same document as the source
    cards under the finished answer.
    """
    with _chat_lock:
        return JSONResponse({"cards": list(_chat_cards)})


@app.get("/api/chat/config")
def chat_config():
    """Introspection for the chatbot tab: which model answers, over which
    engine pool, with exactly which prompts — nothing hidden."""
    activity = _pi.get_activity()
    defs = _module_defaults()
    cfg = app_config.describe()
    return JSONResponse({
        "model": _pi.MODEL,
        "retrieval_model": _pi.MODEL,
        "synthesis_model": getattr(_pi, "SYNTHESIS_MODEL", _pi.MODEL),
        "temperature": 0,
        "context": cfg,
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
    tags: Optional[list[str]] = None
    selected_answers: Optional[list[str]] = None
    selection_mode: str = "any"
    use_cache: bool = False
    replay_answer: bool = False


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

    stale = _stale_index_response()
    if stale is not None:
        return stale

    # A stop applies to the run it was pressed during, never the next one.
    _chat_cancel.clear()
    _pi.clear_cancel()

    defs = _module_defaults()
    index_mod_name = req.index_module or defs["index"]
    try:
        index_mod = _load_module("index", index_mod_name)
    except Exception as exc:
        return JSONResponse({"error": f"Could not load index module '{index_mod_name}': {exc}"}, status_code=400)

    _set_chat_phase("retrieval", "Reading the document trees…")
    try:
        cache_key = _cache_key_for(query, req.tags, req.selected_answers,
                                   req.selection_mode)
        cached = _cache.get(cache_key) if req.use_cache else None
        if cached is not None:
            results, ctx = _replay(cached)
            budget = cached.budget
            cache_info = {"cached": True, **cached.summary()}
        else:
            results, ctx = _run_retrieval(query, index_mod, tags=req.tags,
                                          selected_answers=req.selected_answers,
                                          selection_mode=req.selection_mode)
            budget = _budget_payload(ctx, results)
            cache_info = {"cached": False, "key": cache_key}

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
                    "image": pin.get("image"),
                    "has_source_pdf": doc_name in _load_sources(),
                })

        answer_text = ""
        sections: dict = {}
        # Synthesis is re-run live by default even on a cache hit: the point of
        # the cache is to iterate on the synthesis prompt against a FROZEN node
        # set, which only works if the answer is recomputed.
        if cached is not None and req.replay_answer and cached.answer:
            answer_text = cached.answer.get("content", "")
            sections = {k: v for k, v in cached.answer.items() if k != "content"}
        elif sources:
            # The deck needs the full passages — excerpt, page, pin — to show
            # the same highlight and open the same document the finished
            # answer's source cards do. Those are too big for a 250ms poll,
            # so the poll only says they EXIST and the UI fetches them once.
            manifest = _manifest()
            with _chat_lock:
                _chat_cards.clear()
                _chat_cards.extend({**s_, "folder": _folder_of(manifest.get(s_["doc"]))}
                                   for s_ in sources)
            _set_chat_phase("synthesis",
                            f"Composing an answer from {len(sources)} passages…",
                            cards_ready=len(sources))
            passages = "\n\n".join(
                f"[{s['n']}] {s['doc'].replace('_', ' ')} › {s['breadcrumb'] or s['title']}\n{s['excerpt'] or '(no content)'}"
                for s in sources
            )
            prompt = (
                "You are a clinical knowledge assistant. Answer the question using ONLY the provided source passages. "
                "Every claim must cite its passage inline as [n] (e.g. [1] or [2][3]). Never invent a source number.\n\n"
                f"Question: {query}\n\n"
                f"Source passages:\n{passages}\n\n"
                "Respond in EXACTLY this format (keep the labels, fill in the text; each section is 1-3 sentences of plain text with inline [n] citations):\n\n"
                "SHORT_ANSWER: <the direct answer to the question>\n"
                "RECOMMENDED_ACTION: <what the clinician should do>\n"
                "RATIONALE: <why, grounded in the cited passages>\n"
                "LIMITATIONS: <what the sources do not cover, or uncertainty>\n\n"
                "If the passages cannot answer the question, say so in SHORT_ANSWER and leave the other sections brief.\n"
                "Do not use markdown headers, bullet points, bolding, or lists. Write ONLY plain text under each of the four labels."
            )
            client = _pi.make_client(_pi.OLLAMA_URLS[0])
            # Streamed so a cancel can land mid-answer: breaking out of the
            # iterator closes the connection and Ollama stops generating.
            # (The UI still receives the answer in one piece — this is about
            # being able to STOP, not about showing tokens as they arrive.)
            chunks, prompt_eval = [], 0
            for part in client.chat(
                model=getattr(_pi, "SYNTHESIS_MODEL", _pi.MODEL),
                messages=[{"role": "user", "content": prompt}],
                options=app_config.chat_options("agent"),
                keep_alive=app_config.keep_alive(),
                stream=True,
            ):
                if _chat_cancel.is_set():
                    raise ChatCancelled()
                chunks.append((part.get("message") or {}).get("content") or "")
                prompt_eval = int(part.get("prompt_eval_count") or prompt_eval)
            app_tokens.observe(prompt, prompt_eval)
            answer_text = "".join(chunks)
            # Strip thought blocks
            answer_text = re.sub(r"<thought>.*?(</thought>|$)", "", answer_text, flags=re.S).strip()
            sections = _parse_answer_sections(answer_text)

        if cached is None:
            _cache.put(cache_key, query=query, results=results,
                       events=ctx.events_snapshot(), budget=budget,
                       answer={"content": answer_text, **sections},
                       tags=req.tags, selected_answers=req.selected_answers,
                       index_dir=_index_dir())

        cited = set(int(n) for n in re.findall(r"\[(\d+)\]", answer_text))
        valid_cited = {n for n in cited if 1 <= n <= len(sources)}
        if not sources:
            status = "insufficient_evidence"
            summary = "No passage in the library was judged relevant to this question."
            # With the pre-filter on, "nothing found" may be the filter's doing
            # rather than the library's. Saying which one it was is the
            # difference between a dead end and an obvious next step.
            picked = req.selected_answers or []
            if picked:
                mode = app_choices.normalise_mode(req.selection_mode)
                summary += (
                    f" Your {len(picked)} pre-filter answer"
                    f"{'' if len(picked) == 1 else 's'} narrowed the search"
                    f"{' hard (Fast mode)' if mode == app_choices.ALL else ''}"
                    f" — clearing them searches the whole library.")
        elif valid_cited:
            status = "grounded"
            summary = (f"{len(valid_cited)} of {len(sources)} retrieved passages are "
                       f"cited inline; every source below was selected with a verbatim quote.")
        else:
            status = "partially_grounded"
            summary = (f"{len(sources)} passages were retrieved, but the answer text "
                       f"carries no inline citations — verify against the sources below.")

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
            "budget": budget,
            "choices": _choice_payload(results, req.selected_answers,
                                       req.selection_mode),
            "cache": cache_info,
            "run": {
                "query": query,
                "results": results,
                "budget": budget,
                "test_result": None,
                "pipeline": {
                    "ingest": defs["ingest"],
                    "index":  index_mod_name,
                    "query":  defs["query"],
                },
            },
        })
    except _pi.Cancelled:
        _set_chat_phase("cancelled", "Stopped")
        return JSONResponse({"cancelled": True, "query": query}, status_code=499)
    except ChatCancelled:
        # Not an error: the user asked for this, so it gets its own status
        # rather than a red failure the UI has to explain away.
        _set_chat_phase("cancelled", "Stopped")
        return JSONResponse({"cancelled": True, "query": query}, status_code=499)
    except Exception as exc:
        _set_chat_phase("error", str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        with _chat_lock:
            if _chat_phase["phase"] not in ("error", "cancelled"):
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
