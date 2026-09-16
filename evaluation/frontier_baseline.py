"""
Upper bound: a frontier model with the whole corpus in context.

Same task as the system, not a different one: every chunk is labelled with
its id, and the model is asked to name the chunk ids it would cite — NOT to
answer the question. Otherwise this measures reading comprehension and the
numbers are not comparable with retrieval.

Prompt caching: the corpus goes in the system prompt behind a cache
breakpoint (1-hour TTL); only the question varies after it. The prefix must
be byte-identical across every call, so nothing in it carries a timestamp
or a per-query id. Cache hits are VERIFIED from usage.cache_read_input_tokens
on every response rather than assumed — the run aborts after repeated
misses, because an uncached run of 200 queries over a 200k-token corpus is
the $200 mistake this guard exists for.

The default is a dry run: it builds the prefix, counts tokens and prints an
estimated cost. Nothing is sent to the model until --run is given.

    python3 -m evaluation.frontier_baseline --queries evaluation/queries/queries.csv \
        --out evaluation/results/frontier            # dry run: tokens + cost
    python3 -m evaluation.frontier_baseline ... --run # actually run it

Needs the `anthropic` package and credentials (ANTHROPIC_API_KEY, or an
`ant auth login` profile). Resumable: ids already in frontier.jsonl are
skipped, and the prefix hash is checked against the manifest.

Question-generation leakage, recorded here because it bears on this
comparison: questions written from the chunk text share its vocabulary,
which inflates BM25 and therefore the system, while leaving this full-text
baseline unaffected. The bias runs in the system's favour.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from evaluation.common import (append_jsonl, load_corpus_leaves, load_queries,
                               read_jsonl)

DEFAULT_MODEL = "claude-opus-5"

# Anthropic first-party rates, USD per million tokens, from the claude-api
# skill's cached table (2026-06-24). Printed by the dry run so a stale price
# is visible rather than silently wrong.
PRICES = {
    "claude-opus-5":   {"in": 5.00, "out": 25.00},
    "claude-sonnet-5": {"in": 2.00, "out": 10.00},
}
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10

INSTRUCTIONS = """\
You are evaluating a document retrieval system. Below is an entire corpus of
clinical guideline documents, split into chunks. Every chunk is wrapped in a
<chunk> element with an id attribute of the form "document/section-id".

You will be given one question. Your task is to name the chunk ids you would
cite to answer it — the passages whose text contains the answer. Do NOT
answer the question. Cite every chunk that directly supports an answer and
nothing that merely relates to the topic. If no chunk contains the answer,
return an empty list.
"""


def build_corpus_prefix(leaves) -> str:
    """Deterministic: sorted documents, tree order, no timestamps or ids
    that vary between calls."""
    parts = ["<corpus>"]
    current_doc = None
    for leaf in leaves:
        if leaf.doc != current_doc:
            if current_doc is not None:
                parts.append("</document>")
            parts.append(f'<document name="{leaf.doc}">')
            current_doc = leaf.doc
        parts.append(f'<chunk id="{leaf.chunk_id}" title="{_attr(leaf.breadcrumb or leaf.title)}">')
        parts.append(leaf.content.strip())
        parts.append("</chunk>")
    if current_doc is not None:
        parts.append("</document>")
    parts.append("</corpus>")
    return "\n".join(parts)


def _attr(text: str) -> str:
    return (text or "").replace('"', "'").replace("<", "‹").replace(">", "›")


def system_blocks(prefix: str) -> list[dict]:
    return [
        {"type": "text", "text": INSTRUCTIONS},
        {"type": "text", "text": prefix,
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]


def user_message(question: str) -> str:
    return (f"Question: {question}\n\n"
            "Return the chunk ids you would cite, as they appear in the id attributes.")


def estimate_cost(model: str, prefix_tokens: int, n_queries: int,
                  query_tokens: int = 80, output_tokens: int = 300) -> dict:
    p = PRICES.get(model)
    if p is None:
        return {"model": model, "note": "no price on file for this model"}
    first = prefix_tokens * CACHE_WRITE_MULT * p["in"] / 1e6
    reads = max(0, n_queries - 1) * prefix_tokens * CACHE_READ_MULT * p["in"] / 1e6
    variable = n_queries * (query_tokens * p["in"] + output_tokens * p["out"]) / 1e6
    return {"model": model, "prices_per_mtok": p, "prefix_tokens": prefix_tokens,
            "n_queries": n_queries, "cache_write_usd": round(first, 2),
            "cache_reads_usd": round(reads, 2), "variable_usd": round(variable, 2),
            "total_usd": round(first + reads + variable, 2),
            "note": "assumes one cache write; a >1h pause re-warms the cache once more"}


def parse_ids_from_text(text: str, known: set[str]) -> list[str]:
    found = re.findall(r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", text or "")
    seen, out = set(), []
    for f in found:
        if f in known and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="A,B")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="high", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--run", action="store_true", help="actually call the API (default: dry run)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-cache-misses", type=int, default=3,
                    help="abort after this many consecutive responses with no cache read")
    args = ap.parse_args(argv)

    leaves = load_corpus_leaves(Path(args.index_dir) if args.index_dir else None)
    if not leaves:
        print("no leaves in the index", file=sys.stderr)
        return 1
    known = {l.chunk_id for l in leaves}
    queries = load_queries(Path(args.queries), sets=args.sets.split(","))
    prefix = build_corpus_prefix(leaves)
    prefix_sha = hashlib.sha256(prefix.encode("utf-8")).hexdigest()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus_prefix.txt").write_text(prefix, encoding="utf-8")

    try:
        import anthropic
    except ImportError:
        print("the `anthropic` package is not installed: pip install -r requirements-eval.txt",
              file=sys.stderr)
        return 1
    client = anthropic.Anthropic()

    system = system_blocks(prefix)
    counted = client.messages.count_tokens(
        model=args.model, system=system,
        messages=[{"role": "user", "content": user_message(queries[0].question if queries else "x")}])
    prefix_tokens = int(counted.input_tokens)
    estimate = estimate_cost(args.model, prefix_tokens, len(queries))
    print(f"corpus: {len(leaves)} chunks, prefix sha256 {prefix_sha[:16]}, "
          f"~{prefix_tokens:,} input tokens per call")
    print("estimated cost: " + json.dumps(estimate))
    if not args.run:
        print("dry run — pass --run to execute")
        return 0

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("prefix_sha256") and manifest["prefix_sha256"] != prefix_sha:
        raise SystemExit("the corpus prefix differs from the one this run directory was "
                         "started with (index changed?). Use a new --out directory.")
    manifest.update({"model": args.model, "effort": args.effort, "prefix_sha256": prefix_sha,
                     "prefix_tokens": prefix_tokens, "estimate": estimate,
                     "queries_file": args.queries, "started": manifest.get("started") or time.time()})
    manifest_path.write_text(json.dumps(manifest, indent=2))

    from pydantic import BaseModel

    class CitationChoice(BaseModel):
        cited_chunk_ids: list[str]

    done = {r["id"] for r in read_jsonl(out / "frontier.jsonl")}
    todo = [q for q in queries if q.id not in done]
    if args.limit:
        todo = todo[:args.limit]

    consecutive_misses = 0
    for i, q in enumerate(todo, start=1):
        started = time.perf_counter()
        record = {"id": q.id, "set": q.set, "gold": [q.doc, q.node_id],
                  "gold_chunk_id": f"{q.doc}/{q.node_id}"}
        try:
            response = client.messages.parse(
                model=args.model,
                max_tokens=args.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_message(q.question)}],
                output_config={"effort": args.effort},
                output_format=CitationChoice,
            )
            usage = response.usage
            cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            parsed = getattr(response, "parsed_output", None)
            if parsed is not None:
                raw_ids = list(parsed.cited_chunk_ids)
            else:
                text = "".join(b.text for b in response.content if b.type == "text")
                raw_ids = parse_ids_from_text(text, known)
            cited = [c for c in raw_ids if c in known]
            record.update({
                "cited": cited,
                "invalid_ids": [c for c in raw_ids if c not in known],
                "n_cited": len(cited),
                "gold_cited": record["gold_chunk_id"] in cited,
                "gold_position": (cited.index(record["gold_chunk_id"]) + 1
                                  if record["gold_chunk_id"] in cited else None),
                "stop_reason": response.stop_reason,
                "usage": {"input_tokens": usage.input_tokens,
                          "output_tokens": usage.output_tokens,
                          "cache_read_input_tokens": cache_read,
                          "cache_creation_input_tokens": cache_write},
                "ms": int((time.perf_counter() - started) * 1000),
                "request_id": getattr(response, "_request_id", None),
            })
            if response.stop_reason == "refusal":
                record["refusal"] = getattr(response, "stop_details", None) and \
                    response.stop_details.__dict__
            # Cache verification. The first call of a run legitimately writes;
            # everything after it must read.
            if i == 1 and not done:
                record["cache"] = "write" if cache_write else "none"
                consecutive_misses = 0 if cache_write else 1
            else:
                record["cache"] = "hit" if cache_read else "MISS"
                consecutive_misses = 0 if cache_read else consecutive_misses + 1
        except Exception as exc:  # noqa: BLE001 — record and continue
            record.update({"error": f"{type(exc).__name__}: {exc}",
                           "ms": int((time.perf_counter() - started) * 1000)})
        append_jsonl(out / "frontier.jsonl", record)
        print(f"[{i}/{len(todo)}] {q.id} gold_cited={record.get('gold_cited')} "
              f"n={record.get('n_cited')} cache={record.get('cache')} "
              f"{record.get('error', '')}", file=sys.stderr, flush=True)
        if consecutive_misses >= args.max_cache_misses:
            print(f"{consecutive_misses} consecutive responses read nothing from cache — "
                  "aborting. Check the prefix is byte-identical and long enough to cache.",
                  file=sys.stderr)
            return 2

    records = [r for r in read_jsonl(out / "frontier.jsonl") if not r.get("error")]
    if records:
        hits = sum(1 for r in records if r.get("gold_cited"))
        misses = sum(1 for r in records if r.get("cache") == "MISS")
        print(f"gold cited in {hits}/{len(records)} ({hits / len(records):.1%}); "
              f"mean cited {sum(r['n_cited'] for r in records) / len(records):.2f}; "
              f"cache misses {misses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
