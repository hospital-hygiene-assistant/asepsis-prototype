"""
Debug-only retrieval cache.

Replays a previously-run query's retrieval exactly, without re-running pruning
or leaf evaluation. This is a development aid — it makes iterating on the
synthesis prompt against a frozen node set fast — and is NOT a production
retrieval cache: there is no similarity matching, only exact-key replay.

The key includes an index fingerprint. That is the part that matters: without
it, re-ingesting or re-indexing would keep replaying stale results, and you
would spend an afternoon debugging a "regression" that is really a cache hit
from before the change.

Replays also restore the run's live-event snapshot. Otherwise a cache hit
renders an empty treemap, because starting a run clears every event set.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config as app_config

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".debug_cache"
CACHE_VERSION = 1


def normalise_query(query: str) -> str:
    """Whitespace and case folded, so trivial retyping still hits."""
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def index_fingerprint(index_dir: Path) -> str:
    """A hash over every index file's content.

    Cheap (indexes are small) and exact — mtime would false-positive on a
    no-op rebuild and, worse, false-negative on a same-second edit.
    """
    h = hashlib.sha256()
    index_dir = Path(index_dir)
    if not index_dir.exists():
        return "empty"
    for path in sorted(index_dir.glob("*.json")):
        try:
            h.update(path.name.encode("utf-8"))
            h.update(hashlib.sha256(path.read_bytes()).digest())
        except OSError:
            continue
    return h.hexdigest()[:32]


def make_key(
    query: str,
    index_dir: Path,
    *,
    tags: Optional[list[str]] = None,
    selected_answers: Optional[list[str]] = None,
    retrieval_model: str = "",
    synthesis_model: str = "",
) -> str:
    payload = {
        "v": CACHE_VERSION,
        "query": normalise_query(query),
        "index": index_fingerprint(index_dir),
        "tags": sorted(tags or []),
        "answers": sorted(selected_answers or []),
        "retrieval_model": retrieval_model,
        "synthesis_model": synthesis_model,
        # Any prompt change alters what retrieval does, so it must alter the key.
        "prompts": {k: v for k, v in sorted(app_config.PROMPT_VERSIONS.items())},
        "agent_ctx": app_config.runtime().resolved_agent_ctx(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:40]


@dataclass
class CacheEntry:
    key: str
    query: str
    cached_at: float
    results: dict
    events: dict
    budget: dict
    answer: Optional[dict] = None
    tags: Optional[list] = None
    selected_answers: Optional[list] = None
    # The corpus this run was made against. Compared directly when listing
    # replayable questions — reconstructing the key from ambient config
    # instead would drift the moment any unrelated setting changed.
    fingerprint: str = ""

    @property
    def node_count(self) -> int:
        return sum(len(d.get("retrieved_ids") or []) for d in self.results.values())

    def summary(self) -> dict:
        return {
            "key": self.key,
            "query": self.query,
            "cached_at": self.cached_at,
            "age_seconds": max(0, int(time.time() - self.cached_at)),
            "node_count": self.node_count,
            "doc_count": len(self.results),
            "has_answer": self.answer is not None,
            "tags": self.tags or [],
            "selected_answers": self.selected_answers or [],
        }


class DebugCache:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.dir = Path(cache_dir or CACHE_DIR)

    @property
    def enabled(self) -> bool:
        return bool(app_config.runtime().debug_cache_enabled)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Optional[CacheEntry]:
        if not self.enabled:
            return None
        try:
            raw = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return CacheEntry(**raw)
        except TypeError:
            return None

    def put(self, key: str, *, query: str, results: dict, events: dict,
            budget: dict, answer: Optional[dict] = None,
            tags: Optional[list] = None,
            selected_answers: Optional[list] = None,
            index_dir: Optional[Path] = None) -> Optional[CacheEntry]:
        if not self.enabled:
            return None
        entry = CacheEntry(key=key, query=query, cached_at=time.time(),
                           results=results, events=events, budget=budget,
                           answer=answer, tags=tags,
                           selected_answers=selected_answers,
                           fingerprint=(index_fingerprint(index_dir)
                                        if index_dir is not None else ""))
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._path(key).write_text(
                json.dumps(entry.__dict__, ensure_ascii=False), encoding="utf-8")
        except OSError:
            return None
        return entry

    def entries(self, index_dir: Optional[Path] = None) -> list[CacheEntry]:
        """Every cached run, newest first.

        When `index_dir` is given, only entries whose key still matches the
        current corpus are returned — a stale question must never be offered
        as a replayable one.
        """
        if not self.enabled or not self.dir.exists():
            return []
        out: list[CacheEntry] = []
        for path in self.dir.glob("*.json"):
            try:
                entry = CacheEntry(**json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if index_dir is not None and entry.fingerprint:
                if entry.fingerprint != index_fingerprint(index_dir):
                    continue
            out.append(entry)
        out.sort(key=lambda e: e.cached_at, reverse=True)
        return out

    def clear(self) -> int:
        if not self.dir.exists():
            return 0
        n = len(list(self.dir.glob("*.json")))
        shutil.rmtree(self.dir, ignore_errors=True)
        return n
