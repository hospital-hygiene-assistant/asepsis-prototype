"""Asepsis backend — the HTTP API over pageindex retrieval.

Entrypoint. Assembles the app from api/; each router owns one concern.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# The repo root holds pageindex, paths and modules/, which api/ imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import console
from api.config import CORS_ORIGINS, SERVE_UI
from api.ollama_pool import shutdown_pool, start_configured_instances
from api.routers import chat, documents, ingest, retrieval, status

PORT = 8765


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Bring the Ollama pool up, and take down what we started.

    Importing this module stays side-effect free so tests can load it without
    spawning processes.
    """
    start_configured_instances()
    yield
    shutdown_pool()


app = FastAPI(title="Asepsis Prototype", lifespan=lifespan)

# A same-origin deployment needs no CORS at all.
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware, allow_origins=CORS_ORIGINS,
        allow_methods=["*"], allow_headers=["*"],
    )


@app.middleware("http")
async def _no_cache(request, call_next):
    """Keep the Tauri webview off stale console assets.

    It loads the console from this server and does no hot-reload, so it would
    otherwise serve main.js and index.html from its own cache across launches.
    """
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


for router in (status.router, documents.router, ingest.router,
               retrieval.router, chat.router, console.router):
    app.include_router(router)

if SERVE_UI and console.UI_DIR.is_dir():
    # Mount only when it is there: check_dir=False skips the check at import but
    # StaticFiles re-checks per request and raises, so a missing ui/ would 500
    # rather than 404.
    app.mount("/static", StaticFiles(directory=console.UI_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
