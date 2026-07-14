"""The vanilla dev console.

Optional: ASEPSIS_SERVE_UI=0 runs a bare API, and a missing ui/ directory
degrades to 404 here rather than taking the whole process down.
"""

import re
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from .config import SERVE_UI

router = APIRouter()

UI_DIR = Path(__file__).resolve().parents[1] / "ui"


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


@router.get("/")
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
