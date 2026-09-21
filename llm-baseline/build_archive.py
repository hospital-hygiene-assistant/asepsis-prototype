"""
Consolidate a baseline run into one self-contained JSON archive.

Everything needed to understand or audit the run, without the repository:
the exact cached context (stored ONCE, not repeated per query), the full
model settings, and for every query the question, the raw reply, the parsed
passages, the scoring, the token breakdown, the cache behaviour, the cost and
the latency.

    python3 llm-baseline/build_archive.py
    python3 llm-baseline/build_archive.py --no-corpus   # omit the 2 MB context

The corpus is the bulk of the file. It is included by default because the
point of the archive is to be self-contained: the numbers mean nothing
without the exact bytes the model was shown, and that text is what the cache
key is a hash of.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def percentile(values: list, p: float):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    import math
    k = max(0, min(len(vals) - 1, math.ceil(p / 100.0 * len(vals)) - 1))
    return vals[k]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(HERE / "results" / "results.jsonl"))
    ap.add_argument("--out", default=str(HERE / "results" / "baseline_archive.json"))
    ap.add_argument("--no-corpus", action="store_true")
    args = ap.parse_args(argv)

    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    provider_name = cfg.get("provider")
    provider = (cfg.get("providers") or {}).get(provider_name) or {}
    corpus = (HERE / "corpus_context.txt").read_text(encoding="utf-8")
    instructions = (HERE / "instructions.txt").read_text(encoding="utf-8")
    prefix = f"{corpus.rstrip()}\n\n{'=' * 78}\n\n{instructions.rstrip()}"
    manifest = json.loads((HERE / "corpus_manifest.json").read_text(encoding="utf-8"))
    gold_map = json.loads((HERE / "gold_map.json").read_text(encoding="utf-8"))

    records = [json.loads(l) for l in
               Path(args.results).read_text(encoding="utf-8").splitlines() if l.strip()]
    records = list({r["id"]: r for r in records}.values())   # last write wins
    records.sort(key=lambda r: r["id"])
    ok = [r for r in records if not r.get("error")]

    prompts_index = {}
    idx_path = HERE / "prompts" / "index.json"
    if idx_path.exists():
        prompts_index = {p["id"]: p for p in
                         json.loads(idx_path.read_text(encoding="utf-8"))["prompts"]}

    queries = []
    for r in records:
        meta = prompts_index.get(r["id"], {})
        gold = gold_map.get(r["id"], {})
        queries.append({
            "id": r["id"],
            "set": r.get("set"),
            "pair_id": r.get("pair_id"),
            "question": r.get("question") or meta.get("question"),
            "prompt_suffix": (f"QUESTION: {r.get('question')}\n\n"
                              "Reply with the JSON object only."),
            "gold": {
                "doc": gold.get("doc"),
                "label": gold.get("label"),
                "gold_node_id": gold.get("gold_node_id"),
                "target_leaf": gold.get("target_leaf"),
                "chunk_id": gold.get("chunk_id"),
                "section_level": gold.get("section_level"),
            },
            "response": {
                "raw_text": r.get("raw_text"),
                "status": r.get("status"),
                "incomplete_reason": r.get("incomplete_reason"),
                "parse_error": r.get("parse_error"),
                "passages": r.get("passages"),
            },
            "scoring": {
                "gold_retrieved": r.get("gold_retrieved"),
                "gold_rank": r.get("gold_rank"),
                "n_returned": r.get("n_returned"),
                "n_same_document": r.get("n_same_document"),
                "n_unknown_ids": r.get("n_unknown_ids"),
                "n_quotes_not_verbatim": r.get("n_quotes_not_verbatim"),
            },
            "usage": r.get("usage"),
            "tokens": r.get("tokens"),
            "cost_usd": r.get("cost"),
            "latency_ms": r.get("latency_ms"),
            "cache_variant": r.get("cache_variant"),
            "prefix_sha256": r.get("prefix_sha256"),
            "timestamp": r.get("ts"),
            "error": r.get("error"),
        })

    lat = [r["latency_ms"] for r in ok]
    ret = [r["n_returned"] for r in ok]
    hits = [r for r in ok if r.get("gold_retrieved")]
    rank1 = [r for r in hits if r.get("gold_rank") == 1]
    section_level = [r for r in ok if (gold_map.get(r["id"]) or {}).get("section_level")]
    section_hits = [r for r in section_level if r.get("gold_retrieved")]

    totals = {
        "queries_total": len(records),
        "queries_ok": len(ok),
        "queries_failed": len(records) - len(ok),
        "cost_usd_total": round(sum((r.get("cost") or {}).get("total_usd", 0) for r in ok), 4),
        "cost_usd_cache_write": round(
            sum((r.get("cost") or {}).get("cache_write_usd", 0) for r in ok), 4),
        "cost_usd_cache_read": round(
            sum((r.get("cost") or {}).get("cache_read_usd", 0) for r in ok), 4),
        "cost_usd_output": round(sum((r.get("cost") or {}).get("output_usd", 0) for r in ok), 4),
        "tokens_input_total": sum(r["tokens"]["input_tokens"] for r in ok),
        "tokens_cached_total": sum(r["tokens"]["cached_tokens"] for r in ok),
        "tokens_cache_write_total": sum(r["tokens"]["cache_write_tokens"] for r in ok),
        "tokens_output_total": sum(r["tokens"]["output_tokens"] for r in ok),
        "tokens_reasoning_total": sum(r["tokens"]["reasoning_tokens"] for r in ok),
        "cache_hits": sum(1 for r in ok if r["tokens"]["cached_tokens"] > 0),
        "cache_writes": sum(1 for r in ok if r["tokens"]["cache_write_tokens"] > 0),
        "latency_ms_p50": percentile(lat, 50),
        "latency_ms_p95": percentile(lat, 95),
        "latency_ms_mean": round(statistics.mean(lat), 1) if lat else None,
        "wall_clock_s_if_sequential": round(sum(lat) / 1000, 1) if lat else None,
        "passages_returned_mean": round(statistics.mean(ret), 2) if ret else None,
        "passages_returned_median": statistics.median(ret) if ret else None,
        "passages_returned_max": max(ret) if ret else None,
        "gold_retrieved": len(hits),
        "gold_recall": round(len(hits) / len(ok), 4) if ok else None,
        "gold_at_rank_1": len(rank1),
        "gold_recall_at_1": round(len(rank1) / len(ok), 4) if ok else None,
        "gold_recall_section_level": (round(len(section_hits) / len(section_level), 4)
                                      if section_level else None),
        "section_level_queries": len(section_level),
        "quotes_not_verbatim": sum(r.get("n_quotes_not_verbatim", 0) for r in ok),
        "unknown_ids_returned": sum(r.get("n_unknown_ids", 0) for r in ok),
        "parse_errors": sum(1 for r in ok if r.get("parse_error")),
    }

    archive = {
        "archive_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what_this_is": (
            "Whole-corpus LLM retrieval baseline: a frontier model is given the "
            "entire knowledge base in one cached context and asked which passages "
            "it would cite for each question. It is set the same task as the "
            "system under evaluation (retrieval, not question answering), so the "
            "comparison is retrieval against retrieval."),
        "run": {
            "provider": provider_name,
            "model": provider.get("model_id"),
            "endpoint": provider.get("base_url"),
            "api": "Responses API, POST /responses",
            "settings": {
                "max_output_tokens": cfg.get("max_output_tokens"),
                "reasoning_effort": cfg.get("reasoning_effort"),
                "prompt_cache": cfg.get("prompt_cache"),
                "prompt_cache_key": cfg.get("prompt_cache_key"),
                "prompt_cache_ttl": cfg.get("prompt_cache_ttl"),
                "prompt_cache_options": {"mode": "explicit",
                                         "ttl": cfg.get("prompt_cache_ttl")},
                "prompt_cache_breakpoint": {"mode": "explicit",
                                            "placed_after": "corpus + instructions"},
                "temperature": "not sent (reasoning model)",
            },
            "pricing_usd_per_mtok": provider.get("pricing_usd_per_mtok"),
            "long_context_threshold_tokens": cfg.get("long_context_threshold_tokens"),
            "queries_csv": cfg.get("queries_csv"),
        },
        "cached_context": {
            "note": ("Stored once. This exact text was the cached prefix of every "
                     "call; only the question varied after the cache breakpoint."),
            "sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
            "chars": len(prefix),
            "tokens_measured": (ok[0]["tokens"]["input_tokens"] - ok[0]["tokens"]["fresh_tokens"]
                                if ok else None),
            "n_passages": manifest["n_chunks"],
            "n_documents": manifest["n_docs"],
            "instructions": instructions,
            "corpus": None if args.no_corpus else corpus,
            "corpus_omitted": bool(args.no_corpus),
        },
        "corpus_manifest": manifest["chunks"],
        "gold_map": gold_map,
        "totals": totals,
        "queries": queries,
    }

    out = Path(args.out)
    out.write_text(json.dumps(archive, indent=1, ensure_ascii=False), encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size_mb:.1f} MB)")
    for k in ("queries_ok", "queries_failed", "cost_usd_total", "cache_hits",
              "cache_writes", "gold_recall", "gold_recall_at_1",
              "passages_returned_mean", "latency_ms_p50", "latency_ms_p95",
              "quotes_not_verbatim", "unknown_ids_returned", "parse_errors"):
        print(f"  {k:28} {totals[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
