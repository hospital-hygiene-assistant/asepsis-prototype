#!/usr/bin/env python3
"""
One-click launcher for the Asepsis Prototype.

  python3 run-tauri.py

Runs the full pipeline (ingest → index) if needed, starts the Python backend
on localhost:8765, then opens a native Tauri window or falls back to the browser.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pageindex.library import (  # noqa: E402
    ExpectedLibraryCorrupt,
    ExpectedLibraryNotBuilt,
    ExpectedLibraryStore,
)
from paths import (  # noqa: E402
    DOCS_DIR,
    KB_DIR,
    LIBRARY_DIR,
    ROOT,
    SOURCES_MANIFEST,
)

TAURI_DIR = ROOT / "tauri-app"
# Both stages go through pipeline.py so the launcher uses the same ingest adapter
# and immutable generation publisher as other local entry points.
PIPELINE  = ROOT / "pipeline.py"
PORT      = 8765
URL       = f"http://127.0.0.1:{PORT}"


# ── dependency check ──────────────────────────────────────────
def _ensure_python_deps():
    try:
        import fastapi, uvicorn  # noqa: F401
    except ImportError:
        print("  Installing Python dependencies…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")]
        )


# ── pipeline ──────────────────────────────────────────────────
def _run_pipeline_if_needed():
    docs = list(DOCS_DIR.glob("*.md")) if DOCS_DIR.exists() else []
    if not docs:
        print("  No docs found in docs/ — skipping pipeline.")
        return

    # Ingest: run if knowledge_base is missing or stale
    kb_paths = {path.stem: path for path in KB_DIR.glob("*.md")}
    kb_files = set(kb_paths)
    doc_stems = set(p.stem for p in docs)
    needs_ingest = not kb_files >= doc_stems or any(
        source.stat().st_mtime_ns > kb_paths[source.stem].stat().st_mtime_ns
        for source in docs
        if source.stem in kb_paths
    )
    if needs_ingest:
        print("  Running ingest…")
        subprocess.check_call([sys.executable, str(PIPELINE), "ingest"], cwd=ROOT)
        kb_paths = {path.stem: path for path in KB_DIR.glob("*.md")}
        kb_files = set(kb_paths)

    # Library: rebuild if its complete canonical Markdown set has changed.
    library_store = ExpectedLibraryStore(LIBRARY_DIR)
    try:
        snapshot = library_store.open_current()
        published = {
            document.document_id: document.canonical_markdown
            for document in snapshot.documents
        }
    except (ExpectedLibraryNotBuilt, ExpectedLibraryCorrupt):
        published = {}
    library_is_stale = set(published) != kb_files or any(
        published.get(stem) != path.read_text(encoding="utf-8")
        for stem, path in kb_paths.items()
    )
    library_is_stale = library_is_stale or (
        SOURCES_MANIFEST.is_file()
        and library_store.current.is_file()
        and SOURCES_MANIFEST.stat().st_mtime_ns
        > library_store.current.stat().st_mtime_ns
    )
    if library_is_stale:
        print("  Building index…")
        subprocess.check_call([sys.executable, str(PIPELINE), "index"], cwd=ROOT)


# ── server readiness ──────────────────────────────────────────
def _wait_for_server(timeout: int = 20) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(URL, timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False


# ── Tauri detection ───────────────────────────────────────────
def _has_tauri() -> bool:
    if not shutil.which("cargo"):
        return False
    r = subprocess.run(
        ["cargo", "tauri", "--version"],
        capture_output=True,
        cwd=TAURI_DIR,
    )
    return r.returncode == 0


# ── main ──────────────────────────────────────────────────────
def _print_ollama_tip():
    urls = os.environ.get("OLLAMA_URLS", "")
    n = len([u for u in urls.split(",") if u.strip()]) if urls else 1
    if n > 1:
        print(f"  Ollama pool  → {n} instances ({urls})")
    else:
        print("  Ollama tip   → For faster retrieval run multiple Ollama instances and set:")
        print("                   OLLAMA_URLS=http://localhost:11434,http://localhost:11435")
        print("                 Start extras with:  OLLAMA_HOST=0.0.0.0:11435 ollama serve")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Launch the Asepsis Prototype")
    parser.add_argument("--ollama-instances", type=int, default=1,
                        help="Number of Ollama instances to use for parallel retrieval (default: 1)")
    args = parser.parse_args()

    print("\nAsepsis Prototype — starting up")
    print("─" * 40)

    _ensure_python_deps()
    _print_ollama_tip()
    _run_pipeline_if_needed()

    # Launch Python backend
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OLLAMA_INSTANCES": str(args.ollama_instances)}
    server = subprocess.Popen(
        [sys.executable, str(TAURI_DIR / "server.py")],
        env=env,
    )

    print(f"  Backend  → {URL}")
    if not _wait_for_server():
        print("ERROR: backend did not start within 20 s.")
        server.terminate()
        sys.exit(1)
    print("  Backend ready ✓")

    try:
        if _has_tauri():
            print("  Opening Tauri window…")
            print("  (first run compiles Rust — may take a minute)\n")
            subprocess.run(["cargo", "tauri", "dev"], cwd=TAURI_DIR)
        else:
            print("  tauri-cli not found — opening in system browser.")
            print(f"  App: {URL}")
            print("  Press Ctrl+C to stop.\n")
            import webbrowser
            webbrowser.open(URL)
            if hasattr(signal, "pause"):
                signal.pause()
            else:
                while True:
                    time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  Shutting down…")
        server.terminate()
        try:
            server.wait(timeout=4)
        except subprocess.TimeoutExpired:
            server.kill()
        print("  Done.")


if __name__ == "__main__":
    main()
