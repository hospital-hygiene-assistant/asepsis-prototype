"""
Pluggable asset-captioning backends for the BetterIngest PDF ingest module.

Captions (not raw crops) are what feed the RAG decision step, so every
figure/table crop gets a retrieval-oriented description.  Medical content
should not leave the machine by default, so the default backend is a local
Ollama vision model.  Backends:

  ollama          (default) local vision model via the Ollama HTTP API.
                  Model from $BETTERINGEST_CAPTION_MODEL, else gemma4:e2b
                  (vision-capable and already used by astepsis retrieval).
                  The model must actually be pulled — a missing model is a
                  hard error, never a silent downgrade.
  betteringester  BetterIngester's own cached, rate-limited cloud transport
                  (run_pipeline._api_chat).  Needs $BETTERINGESTER_ROOT
                  pointing at a BetterIngester checkout and its .env/API key.
  secure_server   reserved seam for a future local secure-server backend —
                  intentionally not implemented yet.

All backends write through a deterministic on-disk cache keyed by
(model, sha256 of the crop PNG), so re-running ingest on the same folder is
free and reproducible — descriptions are content, not join keys, but stable
re-runs keep knowledge_base/ byte-identical.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

DEFAULT_OLLAMA_MODEL = os.environ.get("BETTERINGEST_CAPTION_MODEL", "gemma4:e2b")
DEFAULT_OLLAMA_URL = os.environ.get("BETTERINGEST_CAPTION_URL",
                                    "http://127.0.0.1:11434")

_cache_lock = threading.Lock()


class CaptioningUnavailable(RuntimeError):
    """A captioning backend is missing a model/dependency.  Callers surface
    this to the user instead of silently downgrading."""


# ── deterministic cache ──────────────────────────────────────────────────────

def _cache_load(cache_file: Path) -> dict:
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _cache_key(model: str, image_path: Path) -> str:
    h = hashlib.sha256(image_path.read_bytes()).hexdigest()
    return f"{model}\x00{h}"


# ── backend: local Ollama vision ─────────────────────────────────────────────

def _ollama_chat_factory(model: str, url: str):
    import ollama
    client = ollama.Client(host=url)

    # Fail loudly if the model isn't pulled (no silent downgrade).
    try:
        pulled = [m.get("model") or m.get("name", "")
                  for m in client.list().get("models", [])]
    except Exception as exc:
        raise CaptioningUnavailable(
            f"Ollama not reachable at {url} for asset captioning: {exc}") from exc
    if not any(p == model or p.split(":")[0] == model for p in pulled):
        raise CaptioningUnavailable(
            f"Vision model '{model}' is not pulled in Ollama ({url}). "
            f"Pulled models: {', '.join(pulled) or '(none)'}. "
            f"Run `ollama pull {model}` or set $BETTERINGEST_CAPTION_MODEL.")

    def chat(messages):
        # Adapt OpenAI-style multimodal messages (text + data-URL image parts,
        # the shape BetterIngest.describe_assets emits) to the Ollama API.
        text_parts, images = [], []
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                text_parts.append(content)
                continue
            for part in content or []:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "image_url":
                    url_ = part["image_url"]["url"]
                    b64 = url_.split("base64,", 1)[1]
                    images.append(base64.b64decode(b64))
        resp = client.chat(model=model, messages=[{
            "role": "user", "content": "\n".join(text_parts),
            "images": images,
        }], options={"temperature": 0})
        return resp["message"]["content"]

    return chat


# ── backend: BetterIngester's cached cloud transport ─────────────────────────

def _betteringester_chat_factory(model: str | None):
    root = os.environ.get("BETTERINGESTER_ROOT", "")
    if not root or not (Path(root) / "run_pipeline.py").exists():
        raise CaptioningUnavailable(
            "The 'betteringester' captioning backend needs $BETTERINGESTER_ROOT "
            "pointing at a BetterIngester checkout (with run_pipeline.py and "
            "its .env API key).")
    if not model:
        raise CaptioningUnavailable(
            "The 'betteringester' backend needs an explicit model name "
            "(e.g. gemma-4-31b-it) — set $BETTERINGEST_CAPTION_MODEL.")

    # run_pipeline imports `bench.*` and `ingest.*` from ITS repo root, and
    # astepsis has an unrelated top-level ingest.py.  Give BetterIngester's
    # root import priority and drop any astepsis-owned 'ingest'/'bench'
    # entries from sys.modules before importing.  (Nothing in the astepsis
    # server imports plain `ingest` — run-tauri runs it as a script — so
    # leaving BetterIngester's package under that name afterwards is safe.)
    astepsis_root = str(Path(__file__).resolve().parents[2])
    for name in list(sys.modules):
        if name == "ingest" or name.startswith(("ingest.", "bench")):
            mod_file = getattr(sys.modules[name], "__file__", "") or ""
            if mod_file.startswith(astepsis_root):
                del sys.modules[name]
    if root not in sys.path:
        sys.path.insert(0, root)
    import run_pipeline  # noqa: E402  (BetterIngester's harness)

    def chat(messages):
        return run_pipeline._api_chat(model, messages, bucket=model)

    return chat


# ── backend registry (the pluggable seam) ────────────────────────────────────

def _secure_server_chat_factory(model: str | None):
    raise CaptioningUnavailable(
        "The 'secure_server' captioning backend is a reserved seam for a "
        "future local secure-server deployment and is not implemented yet.")


BACKENDS = {
    "ollama": lambda model: _ollama_chat_factory(
        model or DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL),
    "betteringester": _betteringester_chat_factory,
    "secure_server": _secure_server_chat_factory,
}


def caption_assets(bi, doc, backend: str = "ollama", model: str | None = None,
                   cache_dir: str | Path = ".betteringest_out") -> None:
    """Fill `doc.assets[*].description` via the chosen backend, write-through
    cached by (model, image sha).  `bi` is the BetterIngest instance (its
    describe_assets carries the prompt).  Mutates `doc` in place."""
    if not doc.assets:
        return
    if backend not in BACKENDS:
        raise CaptioningUnavailable(
            f"Unknown captioning backend '{backend}'. "
            f"Available: {', '.join(BACKENDS)}.")

    effective_model = model or (DEFAULT_OLLAMA_MODEL if backend == "ollama"
                                else os.environ.get("BETTERINGEST_CAPTION_MODEL"))
    cache_file = Path(cache_dir) / "captions.json"
    with _cache_lock:
        cache = _cache_load(cache_file)

    missing = []
    for a in doc.assets:
        key = _cache_key(effective_model or "", Path(a.image))
        if key in cache:
            a.description = cache[key]
        else:
            missing.append((a, key))
    if not missing:
        return

    chat = BACKENDS[backend](model)          # may raise CaptioningUnavailable
    bi.describe_assets(doc, chat=chat)

    with _cache_lock:
        cache = _cache_load(cache_file)      # re-read: parallel writers
        for a, key in missing:
            if a.description:
                cache[key] = a.description
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                              encoding="utf-8")
