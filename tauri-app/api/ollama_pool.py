"""Ollama process pool.

Owns the extra `ollama serve` instances this process starts, and the dedicated
explainer instance. Retrieval clients themselves live in pageindex.
"""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pageindex as _pi
from pageindex.settings import MAX_OLLAMA_INSTANCES, MIN_OLLAMA_INSTANCES


BASE_PORT = 11434


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


CPU_ONLY = _env_flag("ASEPSIS_OLLAMA_CPU_ONLY", True)


def ollama_process_env(*, host: str) -> dict[str, str]:
    """Environment for every Ollama process ASEPSIS itself owns."""
    env = {
        **os.environ,
        "OLLAMA_HOST": host,
        "OLLAMA_NUM_PARALLEL": "1",
    }
    if CPU_ONLY:
        env.update({
            "OLLAMA_VULKAN": "0",
            "GGML_VK_VISIBLE_DEVICES": "-1",
            "CUDA_VISIBLE_DEVICES": "-1",
            "ROCR_VISIBLE_DEVICES": "-1",
        })
    return env


def processor_status(url: str) -> dict:
    """Report whether loaded models prove the requested CPU-only profile."""
    status = {"cpu_only_requested": CPU_ONLY, "verified": None, "models": []}
    try:
        with urllib.request.urlopen(f"{url}/api/ps", timeout=2.0) as response:
            payload = json.loads(response.read())
        models = payload.get("models", [])
        status["models"] = [{
            "name": model.get("name", ""),
            "size_vram": int(model.get("size_vram", 0)),
        } for model in models]
        if models:
            status["verified"] = all(model["size_vram"] == 0 for model in status["models"])
    except Exception as exc:
        status["error"] = str(exc)
    return status

# Instances we started, in port order from BASE_PORT+1. A None entry marks a
# port that was already serving: it is counted, but not ours to stop.
_extra_procs: list[Optional[subprocess.Popen]] = []


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


def ollama_bin() -> Optional[str]:
    """Path to the ollama binary, or None when it is not installed."""
    return shutil.which("ollama")


def set_ollama_instances(n: int) -> dict:
    """Start/stop Ollama instances so exactly n are running. Returns status dict."""
    if isinstance(n, bool) or not MIN_OLLAMA_INSTANCES <= n <= MAX_OLLAMA_INSTANCES:
        raise ValueError(
            f"ollama instance count must be between {MIN_OLLAMA_INSTANCES} "
            f"and {MAX_OLLAMA_INSTANCES}"
        )

    # Drop extras beyond what's needed. A None entry is an instance that was
    # already running when we found it, so there is nothing of ours to stop.
    while len(_extra_procs) > n - 1:
        proc = _extra_procs.pop()
        if proc is None:
            continue
        proc.terminate()
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Start extra instances if needed
    bin_path = ollama_bin()
    errors = []
    while len(_extra_procs) < n - 1:
        port = BASE_PORT + len(_extra_procs) + 1
        url  = f"http://127.0.0.1:{port}"
        if _probe(url):
            # Already running externally — don't adopt it, just note it
            _extra_procs.append(None)  # placeholder
        elif bin_path:
            env = ollama_process_env(host=f"127.0.0.1:{port}")
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
        bin_path = ollama_bin()
        if not bin_path:
            base = f"http://127.0.0.1:{BASE_PORT}"
            return base if _probe(base) else None
        env = ollama_process_env(host=f"127.0.0.1:{EXPLAINER_PORT}")
        proc = subprocess.Popen(
            [bin_path, "serve"], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _explainer_procs.append(proc)
        if _wait_ready(url, timeout=25):
            return url
        base = f"http://127.0.0.1:{BASE_PORT}"
        return base if _probe(base) else None


def shutdown_pool() -> None:
    """Terminate every instance this process started.

    Instances found already running are held as None placeholders and left
    alone: this process does not own them. Safe to call more than once.
    """
    for proc in [*_explainer_procs, *_extra_procs]:
        if proc is None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=4)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _explainer_procs.clear()
    _extra_procs.clear()


def start_configured_instances() -> None:
    """Bring the pool up to $OLLAMA_INSTANCES. Called once at app startup."""
    n = int(os.environ.get("OLLAMA_INSTANCES", "1"))
    if not MIN_OLLAMA_INSTANCES <= n <= MAX_OLLAMA_INSTANCES:
        raise ValueError(
            f"OLLAMA_INSTANCES must be between {MIN_OLLAMA_INSTANCES} "
            f"and {MAX_OLLAMA_INSTANCES}"
        )
    if n > 1:
        set_ollama_instances(n)


# Safety net for exits that bypass the app lifespan.
atexit.register(shutdown_pool)
