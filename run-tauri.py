#!/usr/bin/env python3
"""
One-click launcher for PageIndex Explorer.

  python3 run-tauri.py

Starts the Python backend on localhost:8765, then:
  - opens a native Tauri window  (if cargo + tauri-cli v1 are installed)
  - falls back to the system browser otherwise

Install Tauri CLI v1 once with:
  cargo install tauri-cli --version '^1.0' --locked
"""

import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT      = Path(__file__).parent
TAURI_DIR = ROOT / "tauri-app"
PORT      = 8765
URL       = f"http://127.0.0.1:{PORT}"


# ── dependency check / install ────────────────────────────────
def _ensure_python_deps():
    req = TAURI_DIR / "requirements-tauri.txt"
    try:
        import fastapi, uvicorn  # noqa: F401
    except ImportError:
        print("  Installing Python dependencies...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)]
        )


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
def main():
    print("\nPageIndex Explorer — starting up")
    print("─" * 40)

    _ensure_python_deps()

    # Launch Python backend
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
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
            # Block until user interrupts
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
