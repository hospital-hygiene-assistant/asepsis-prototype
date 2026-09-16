"""
Structured per-run log.

One JSON object per line, appended after every retrieval or chat run, holding
what an evaluation needs and the UI does not: phase timings, the budget, the
mechanical grounding status, the model's raw sufficiency label, every node's
terminal status with its BM25 rank, and the configuration and index the run
was made against.

Off by default. Set ASEPSIS_RUN_LOG=/path/to/runs.jsonl to turn it on — an
environment variable rather than a runtime setting so that a measured run
cannot be switched off from the UI halfway through.

The index fingerprint is recorded on every line for one reason: re-indexing
in the middle of a measured run silently changes the corpus underneath the
later queries, and nothing else would show it.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import config as app_config
import debug_cache

ENV_VAR = "ASEPSIS_RUN_LOG"
_lock = threading.Lock()


def log_path() -> Optional[Path]:
    raw = os.environ.get(ENV_VAR, "").strip()
    return Path(raw) if raw else None


def enabled() -> bool:
    return log_path() is not None


def config_snapshot() -> dict:
    cfg = app_config.runtime()
    return {
        "retrieval_model": cfg.retrieval_model,
        "synthesis_model": cfg.synthesis_model,
        "retrieval_ctx": cfg.resolved_retrieval_ctx(),
        "agent_ctx": cfg.resolved_agent_ctx(),
        "max_leaf_evals": cfg.max_leaf_evals,
        "concurrency_per_instance": cfg.concurrency_per_instance,
        "completeness_check": cfg.completeness_check,
        "debug_cache_enabled": cfg.debug_cache_enabled,
        "prompt_versions": dict(app_config.PROMPT_VERSIONS),
    }


def node_statuses(results: dict) -> dict:
    """{doc: {node_id: {status, bm25_rank, bm25_score, ms, judged_relevant}}}

    Only leaves and judged sections carry an entry; a node absent here was
    never touched by the run (which, for a leaf, cannot happen — every leaf
    ends a run in exactly one terminal status).
    """
    out: dict[str, dict] = {}
    for doc, data in (results or {}).items():
        per = {}
        for node_id, meta in (data.get("node_meta") or {}).items():
            entry = {"status": meta.get("status")}
            for key in ("bm25_rank", "bm25_score", "ms", "judged_relevant"):
                if meta.get(key) is not None:
                    entry[key] = meta[key]
            per[node_id] = entry
        out[doc] = per
    return out


def build_record(*, kind: str, query: str, results: dict, budget: dict,
                 timing: dict, index_dir: Path, run_id: str = "",
                 sources: Optional[list] = None, answer_text: str = "",
                 grounding_status: str = "", judgment: Optional[dict] = None,
                 cited: Optional[list] = None, extra: Optional[dict] = None) -> dict:
    retrieved = [{"doc": s["doc"], "node_id": s["node_id"], "n": s["n"]}
                 for s in (sources or [])]
    by_n = {s["n"]: s for s in (sources or [])}
    cited_nodes = [{"doc": by_n[n]["doc"], "node_id": by_n[n]["node_id"], "n": n}
                   for n in sorted(set(cited or [])) if n in by_n]
    record = {
        "ts": time.time(),
        "kind": kind,
        "run_id": run_id,
        "query": query,
        "timing": dict(timing or {}),
        "budget": {k: v for k, v in (budget or {}).items() if k != "deferred_nodes"},
        "deferred_nodes": [{"doc": d["doc"], "node_id": d["node_id"],
                            "bm25_rank": d.get("bm25_rank")}
                           for d in (budget or {}).get("deferred_nodes", [])],
        "grounding_status": grounding_status,
        "judgment": judgment or {},
        "retrieved": retrieved,
        "cited": cited_nodes,
        "answer_text": answer_text,
        "node_statuses": node_statuses(results),
        "index_fingerprint": debug_cache.index_fingerprint(index_dir),
        "config": config_snapshot(),
    }
    if extra:
        record.update(extra)
    return record


def append(record: dict) -> bool:
    """Append one record. Never raises: a logging failure must not fail a run."""
    path = log_path()
    if path is None:
        return False
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return True
    except Exception as exc:  # noqa: BLE001 — deliberately broad
        print(f"[runlog] could not append to {path}: {exc}", file=sys.stderr)
        return False
