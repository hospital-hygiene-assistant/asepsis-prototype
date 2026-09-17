"""
The evaluation driver: Sets A and B through the real chat endpoint.

    # main arm, production settings
    python3 -m evaluation.run_eval --queries evaluation/queries/queries.csv \
        --out evaluation/results/main

    # raised-cap arm: ONLY the queries whose gold chunk was deferred in main
    python3 -m evaluation.run_eval --queries evaluation/queries/queries.csv \
        --out evaluation/results/raised --arm raised \
        --only-gold-deferred-from evaluation/results/main \
        --agent-ctx 131072 --max-leaf-evals 2000

The backend must already be running (python3 tauri-app/server.py, or the
launcher). The driver never injects the gold chunk, never enables the
retrieval cache, and never passes pre-filter answers; it fails fast if the
server is busy, the index is stale, or the completeness check is off.

Every response is saved whole under raw/<id>.json, and a flat record per
query is appended to records.jsonl. The index fingerprint is re-read after
every query: a change means the corpus was re-indexed underneath the run,
and the driver stops rather than mixing two corpora in one result set.

Resumable: ids already in records.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from evaluation.common import (Query, append_jsonl, derive_record, file_sha256,
                               load_queries, load_records)

DEFAULT_SERVER = "http://127.0.0.1:8791"
DEFERRED_OUTCOMES = {"deferred", "deferred_judged_relevant"}


class Server:
    def __init__(self, base: str, timeout: float = 3600.0):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post(self, path: str, body: dict, timeout: Optional[float] = None) -> tuple[int, dict]:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                body_out = json.loads(exc.read().decode("utf-8"))
            except Exception:
                body_out = {"error": str(exc)}
            return exc.code, body_out

    def fingerprint(self) -> str:
        state = self.get("/api/index/state")
        return state.get("fingerprint") or ""


def preflight(server: Server, allow_busy: bool = False) -> dict:
    """Refuse to start a measured run in a state that would make it not one."""
    problems = []
    config = server.get("/api/config")
    state = server.get("/api/index/state")
    status = server.get("/api/status")

    if state.get("stale"):
        problems.append(f"index is stale for {state.get('stale_docs')} — rebuild first")
    if not state.get("fingerprint"):
        problems.append("server does not report an index fingerprint — "
                        "is this the evaluation branch's server.py?")
    if not config.get("completeness_check", True):
        problems.append("completeness_check is off — the sufficiency judgment "
                        "would never be emitted (POST /api/config completeness_check=true)")
    if status.get("any_busy") and not allow_busy:
        problems.append("an Ollama instance is busy — another run is in flight")
    if config.get("debug_cache_enabled"):
        print("note: debug cache is enabled. It only records runs (use_cache is "
              "never sent), so measurements are unaffected.", file=sys.stderr)
    if problems:
        raise SystemExit("preflight failed:\n  " + "\n  ".join(problems))
    return {"config": config, "index": state, "status": status}


def select_queries(queries: list[Query], only_deferred_from: Optional[Path]) -> list[Query]:
    if only_deferred_from is None:
        return queries
    prior = {r["id"]: r for r in load_records(only_deferred_from)}
    keep = [q for q in queries if prior.get(q.id, {}).get("gold_outcome") in DEFERRED_OUTCOMES]
    print(f"{len(keep)} of {len(queries)} queries had a deferred gold chunk in "
          f"{only_deferred_from}", file=sys.stderr)
    return keep


def run(args) -> int:
    server = Server(args.server, timeout=args.timeout)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)

    queries = load_queries(Path(args.queries), sets=args.sets.split(","))
    queries = select_queries(
        queries, Path(args.only_gold_deferred_from) if args.only_gold_deferred_from else None)
    done = {r["id"] for r in load_records(out)}
    todo = [q for q in queries if q.id not in done]
    if args.limit:
        todo = todo[:args.limit]
    if args.dry_run:
        for q in todo:
            print(f"{q.id}\t{q.set}\t{q.doc}/{q.node_id}\t{q.question[:80]}")
        print(f"{len(todo)} queries would run ({len(done)} already done)")
        return 0
    if not todo:
        print("nothing to do — every selected query already has a record")
        return 0

    pre = preflight(server, allow_busy=args.allow_busy)
    fingerprint = pre["index"]["fingerprint"]

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("index_fingerprint") and manifest["index_fingerprint"] != fingerprint:
        raise SystemExit(
            f"this run directory was started against index {manifest['index_fingerprint']} "
            f"but the server now has {fingerprint}. Start a new --out directory.")

    # Arm settings, applied for the duration of the run and restored after.
    original = {k: pre["config"].get(k) for k in ("agent_ctx", "max_leaf_evals")}
    overrides = {k: v for k, v in (("agent_ctx", args.agent_ctx),
                                   ("max_leaf_evals", args.max_leaf_evals)) if v}
    if overrides:
        code, cfg = server.post("/api/config", overrides, timeout=60)
        if code != 200:
            raise SystemExit(f"could not apply arm settings {overrides}: {cfg}")
        print(f"arm settings applied: agent_ctx={cfg.get('agent_ctx')} "
              f"max_leaf_evals={cfg.get('max_leaf_evals')}", file=sys.stderr)
        pre["config"] = cfg

    manifest.update({
        "arm": args.arm,
        "server": args.server,
        "queries_file": str(args.queries),
        "queries_sha256": file_sha256(Path(args.queries)),
        "sets": args.sets,
        "only_gold_deferred_from": args.only_gold_deferred_from,
        "index_fingerprint": fingerprint,
        "config": pre["config"],
        "started": manifest.get("started") or time.time(),
        "aborted": False,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    aborted = ""
    try:
        for i, q in enumerate(todo, start=1):
            started = time.time()
            code, payload = server.post("/api/chat", {"query": q.question, "use_cache": False})
            wall = time.time() - started
            if code != 200:
                payload = {"error": f"HTTP {code}: {payload.get('error') or payload}"}
            (out / "raw" / f"{q.id}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

            now_fp = server.fingerprint()
            rec = derive_record(q, payload, index_fingerprint=now_fp)
            rec.extra["wall_ms"] = int(wall * 1000)
            rec.extra["arm"] = args.arm
            append_jsonl(out / "records.jsonl", asdict(rec))

            t = rec.timing or {}
            print(f"[{i}/{len(todo)}] {q.id} gold={rec.gold_outcome:<24} "
                  f"label={rec.judgment_label:<12} cell={rec.cell:<22} "
                  f"n={rec.n_retrieved:<3} retr={t.get('retrieval_ms')} "
                  f"synth={t.get('synthesis_ms')} {('ERROR ' + rec.error) if rec.error else ''}",
                  file=sys.stderr, flush=True)

            if now_fp != fingerprint and not args.ignore_reindex:
                aborted = (f"index fingerprint changed during the run "
                           f"({fingerprint} → {now_fp}) after {q.id}; stopping. "
                           f"Records from here on would be against a different corpus.")
                print(aborted, file=sys.stderr)
                break
    finally:
        if overrides:
            restore = {k: v for k, v in original.items() if v}
            server.post("/api/config", restore, timeout=60)
            print(f"arm settings restored: {restore}", file=sys.stderr)
        manifest.update({"finished": time.time(), "aborted": bool(aborted),
                         "abort_reason": aborted,
                         "records": len(load_records(out))})
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    return 2 if aborted else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--sets", default="A,B")
    ap.add_argument("--arm", default="main", help="label recorded on every record")
    ap.add_argument("--agent-ctx", type=int, default=None,
                    help="raised-cap arm: agent context window to apply for the run")
    ap.add_argument("--max-leaf-evals", type=int, default=None,
                    help="raised-cap arm: per-run evaluation cap to apply")
    ap.add_argument("--only-gold-deferred-from", default=None,
                    help="run only queries whose gold was deferred in this earlier run dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600.0, help="per-query HTTP timeout, s")
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--ignore-reindex", action="store_true",
                    help="keep going if the index fingerprint changes (not recommended)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
