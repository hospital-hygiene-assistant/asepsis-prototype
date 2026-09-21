"""
Run the whole-corpus retrieval baseline against Amazon Bedrock.

One call per query. The corpus and the task instructions form a byte-identical
cached prefix; only the question varies after the cache point, so the first
call writes the cache and every later call reads it.

    # one query, to check the call works and what it costs
    python3 llm-baseline/run_baseline.py --test A001

    # the full set, resumable
    python3 llm-baseline/run_baseline.py --all

Nothing is sent until --test or --all is given: the default is a dry run that
assembles the request, reports its size, and stops.

The corpus manifest and the gold map are scoring data and are never part of a
request. They are loaded only afterwards, to check three things per query:
whether THE gold chunk was returned (an exact match against the one leaf the
label marks, never an ancestor rule), whether each returned id exists at all,
and whether each verbatim quote really appears in the passage it was
attributed to.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> dict:
    """Load config and flatten the selected provider's settings onto it.

    The corpus, the prompt and the cache breakpoint are identical across
    providers: OpenAI's Responses API and Bedrock's OpenAI-compatible path
    take the same body. Only the endpoint, the model id, the credential and
    the price list differ, so switching provider is a one-word change rather
    than a second code path.
    """
    cfg = json.loads(path.read_text(encoding="utf-8"))
    name = cfg.get("provider")
    provider = (cfg.get("providers") or {}).get(name)
    if provider:
        for key in ("base_url", "model_id", "auth_env", "pricing_usd_per_mtok"):
            if key in provider:
                cfg[key] = provider[key]
        cfg["provider_name"] = name
    return cfg


def _norm(text: str) -> str:
    """Whitespace- and quote-folded, for comparing a returned quote with the
    source passage. Models reflow long quotes and swap curly quotes for
    straight ones; neither is a fabrication, and counting them as one would
    hide the fabrications that matter."""
    text = (text or "").replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


class Corpus:
    """The chunk manifest plus the passage text, for scoring only."""

    def __init__(self, manifest_path: Path, index_dir: Path, gold_map_path: Path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.sha256 = manifest["sha256"]
        self.by_id = {c["chunk_id"]: c for c in manifest["chunks"]}
        # {query_id: chunk_id} — the ONE chunk that holds the answer. Built and
        # verified by build_gold_map.py against the pins in the markdown.
        self.gold_map = json.loads(gold_map_path.read_text(encoding="utf-8"))
        self.content: dict[str, str] = {}
        import pageindex
        for path in sorted(Path(index_dir).glob("*.json")):
            nodes = [pageindex._node_from_dict(d)
                     for d in pageindex.read_index_file(path)["nodes"]]
            by_node = {n.node_id: n for n in pageindex._collect_leaves(nodes)}
            for chunk in manifest["chunks"]:
                if chunk["doc"] == path.stem and chunk["node_id"] in by_node:
                    self.content[chunk["chunk_id"]] = (
                        by_node[chunk["node_id"]].content or "")

    def gold_chunk(self, query_id: str) -> Optional[str]:
        entry = self.gold_map.get(query_id)
        return entry["chunk_id"] if entry else None

    def is_gold(self, chunk_id: str, query_id: str) -> bool:
        """Exact match against the single gold chunk.

        NOT an ancestor rule. A gold label lives on one pin, and a pin marks
        the text directly beneath it. When that pin sits on an internal
        section, the section's own prose has been lifted into a synthetic
        `<section>-overview` leaf which inherits the pin, so the label belongs
        to the overview leaf alone. The section's other subsections carry
        their own pins and not the label; build_gold_map.py verifies that for
        every section-level gold. Crediting any leaf beneath the section would
        make retrieval look better than it is, by up to 95 chunks for the
        widest gold section in this corpus.
        """
        return chunk_id == self.gold_chunk(query_id)

    def quote_is_verbatim(self, chunk_id: str, quote: str) -> Optional[bool]:
        body = self.content.get(chunk_id)
        if body is None:
            return None
        return _norm(quote) in _norm(body)


def build_request(cfg: dict, question: str) -> tuple[dict, str]:
    """(Responses API body, the cached prefix) for one call.

    Caching for GPT-5.6 Luna is Responses-API-only. Converse can call the model
    but has no cache control for it, which on a 445K-token prefix is the
    difference between roughly $2 and roughly $20 for the 100 queries. So the
    request goes to /openai/v1/responses with an explicit breakpoint at the end
    of the static content.

    `mode: explicit` disables the automatic breakpoint on the latest message,
    so the only cached prefix is the one we mark: corpus plus instructions.
    The question follows it and can vary freely without invalidating anything.
    """
    corpus = (ROOT / cfg["corpus_context"]).read_text(encoding="utf-8")
    instructions = (ROOT / cfg["instructions"]).read_text(encoding="utf-8")
    prefix = f"{corpus.rstrip()}\n\n{'=' * 78}\n\n{instructions.rstrip()}"
    suffix = f"QUESTION: {question}\n\nReply with the JSON object only."

    static_block: dict = {"type": "input_text", "text": prefix}
    if cfg.get("prompt_cache", True):
        static_block["prompt_cache_breakpoint"] = {"mode": "explicit"}

    body: dict = {
        "model": cfg["model_id"],
        "max_output_tokens": cfg.get("max_output_tokens", 8192),
        "input": [
            {"type": "message", "role": "developer", "content": [static_block]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": suffix}]},
        ],
    }
    if cfg.get("prompt_cache", True):
        body["prompt_cache_key"] = cfg.get("prompt_cache_key", "asepsis-baseline")
        body["prompt_cache_options"] = {"mode": "explicit",
                                        "ttl": cfg.get("prompt_cache_ttl", "30m")}
    if cfg.get("reasoning_effort"):
        body["reasoning"] = {"effort": cfg["reasoning_effort"]}
    return body, prefix


def parse_passages(text: str) -> tuple[list, str]:
    """(passages, error). Tolerates a fenced block or prose around the JSON."""
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return [], "no JSON object in the reply"
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return [], f"unparseable JSON: {exc}"
    if isinstance(data, list):
        data = {"passages": data}
    if not isinstance(data, dict) or not isinstance(data.get("passages"), list):
        return [], "no 'passages' list in the reply"
    out = []
    for item in data["passages"]:
        if isinstance(item, str):
            out.append({"id": item, "quote": "", "reason": ""})
        elif isinstance(item, dict):
            out.append({"id": str(item.get("id") or "").strip(),
                        "quote": str(item.get("quote") or ""),
                        "reason": str(item.get("reason") or "")})
    return out, ""


def usage_tokens(usage: dict) -> dict:
    """Flatten the Responses API usage object.

    `cached_tokens` and `cache_write_tokens` are subsets of `input_tokens`, so
    the tokens billed at the full input rate are what is left after removing
    both.
    """
    details = usage.get("input_tokens_details") or {}
    total_in = int(usage.get("input_tokens") or 0)
    cached = int(details.get("cached_tokens") or 0)
    written = int(details.get("cache_write_tokens") or 0)
    return {"input_tokens": total_in, "cached_tokens": cached,
            "cache_write_tokens": written,
            "fresh_tokens": max(0, total_in - cached - written),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "reasoning_tokens": int((usage.get("output_tokens_details") or {})
                                    .get("reasoning_tokens") or 0)}


def cost_usd(cfg: dict, usage: dict) -> Optional[dict]:
    """What this call cost, on the tier its input size puts it in.

    Bedrock prices GPT-5.6 Luna in two tiers, and the boundary matters here:
    above 272K input tokens every rate doubles. This corpus is ~445K, so every
    call lands in the long tier. Reporting the tier makes that visible rather
    than leaving it as a silent 2x.
    """
    table = cfg.get("pricing_usd_per_mtok") or {}
    if not table:
        return None
    t = usage_tokens(usage)
    threshold = cfg.get("long_context_threshold_tokens", 272000)
    tier = "long" if t["input_tokens"] > threshold else "short"
    p = table.get(tier)
    if not p:
        return None
    parts = {
        "fresh_input_usd": t["fresh_tokens"] * p["input"] / 1e6,
        "cache_read_usd": t["cached_tokens"] * p["cache_read"] / 1e6,
        "cache_write_usd": t["cache_write_tokens"] * p["cache_write"] / 1e6,
        "output_usd": t["output_tokens"] * p["output"] / 1e6,
    }
    return {"tier": tier, **{k: round(v, 6) for k, v in parts.items()},
            "total_usd": round(sum(parts.values()), 6)}


def _post(url: str, body: dict, token: str, timeout: float) -> tuple[dict, int]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload, int((time.perf_counter() - started) * 1000)


RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.I)


def _post_with_backoff(url: str, body: dict, token: str, timeout: float,
                       max_retries: int = 8) -> tuple[dict, int]:
    """POST, waiting out 429s rather than recording them as failures.

    Each request here is ~445K tokens against a 500K tokens-per-minute
    ceiling, so roughly one call per minute is the most the account can do.
    OpenAI counts CACHED tokens toward that ceiling too, which is the
    opposite of Bedrock's documented behaviour and the reason this pacing is
    needed at all: caching saves money here, not rate limit.

    A 429 is not a failure to record — it is a "come back shortly", and the
    server says exactly how long. Treating it as a result would burn the
    query and leave a hole in the eval set.
    """
    for attempt in range(max_retries + 1):
        try:
            return _post(url, body, token, timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == max_retries:
                raise
            detail = exc.read().decode("utf-8", "replace")
            hint = RETRY_AFTER_RE.search(detail)
            wait = float(hint.group(1)) + 3.0 if hint else min(60.0, 5 * (attempt + 1))
            print(f"    rate limited; waiting {wait:.0f}s "
                  f"(attempt {attempt + 1}/{max_retries})", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def call_responses(cfg: dict, body: dict) -> tuple[dict, int, str]:
    """POST the Responses API. Returns (response, elapsed ms, variant used).

    Raw HTTP on purpose: the cache fields are extensions, and a typed SDK can
    silently drop an unknown field, which here would mean paying the uncached
    rate while believing otherwise.

    The cache parameters are documented for Bedrock. OpenAI direct documents
    prompt caching but not those exact parameter names, and its API rejects
    unknown parameters outright. So an explicit-breakpoint request that comes
    back as a 400 about an unrecognised field is retried without the cache
    fields, falling back to OpenAI's automatic caching, which applies to any
    prefix over 1,024 tokens. The variant that succeeded is recorded on every
    result, because "cached explicitly" and "cached implicitly" are different
    claims and the writeup should not guess between them.
    """
    env_name = cfg.get("auth_env", "AWS_BEARER_TOKEN_BEDROCK")
    token = os.environ.get(env_name, "")
    if not token:
        raise RuntimeError(f"{env_name} is not set "
                           f"(provider: {cfg.get('provider_name', '?')})")
    url = cfg["base_url"].rstrip("/") + "/responses"
    timeout = cfg.get("read_timeout_s", 900)

    stripped = json.loads(json.dumps(body))
    stripped.pop("prompt_cache_options", None)
    stripped.pop("prompt_cache_key", None)
    for item in stripped.get("input", []):
        for block in item.get("content", []) or []:
            block.pop("prompt_cache_breakpoint", None)

    attempts = [("explicit-cache", body), ("implicit-cache", stripped)]
    last = ""
    for label, candidate in attempts:
        try:
            payload, ms = _post_with_backoff(url, candidate, token, timeout)
            return payload, ms, label
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:600]
            last = f"HTTP {exc.code}: {detail}"
            unknown_param = exc.code == 400 and any(
                k in detail.lower() for k in
                ("unknown", "unrecognized", "unrecognised", "unsupported",
                 "additional properties", "not permitted", "extra fields",
                 "prompt_cache"))
            if not unknown_param:
                raise RuntimeError(last) from None
            print(f"  [{label}] rejected the cache parameters; retrying without them",
                  file=sys.stderr)
    raise RuntimeError(last)


def output_text(payload: dict) -> str:
    """The assistant text, ignoring reasoning items."""
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"]
    chunks = []
    for item in payload.get("output") or []:
        if item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if block.get("type") in ("output_text", "text"):
                chunks.append(block.get("text") or "")
    return "".join(chunks)


def run_one(cfg: dict, corpus: Corpus, row: dict, out_dir: Path) -> dict:
    body, prefix = build_request(cfg, row["question"])
    prefix_sha = hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:16]
    response, elapsed_ms, variant = call_responses(cfg, body)

    text = output_text(response)
    passages, parse_error = parse_passages(text)
    usage = response.get("usage", {}) or {}

    seen, deduped = set(), []
    for p in passages:
        if p["id"] and p["id"] not in seen:
            seen.add(p["id"])
            deduped.append(p)

    for p in deduped:
        chunk = corpus.by_id.get(p["id"])
        p["known_id"] = chunk is not None
        p["doc"] = chunk["doc"] if chunk else None
        p["node_id"] = chunk["node_id"] if chunk else None
        p["is_gold"] = corpus.is_gold(p["id"], row["id"])
        p["quote_verbatim"] = corpus.quote_is_verbatim(p["id"], p["quote"])
        # Landing in the right neighbourhood is not a hit, but it is the most
        # useful thing to know about a miss, so it is recorded separately.
        gold_chunk = corpus.by_id.get(corpus.gold_chunk(row["id"]) or "")
        p["same_document"] = bool(gold_chunk and p.get("doc") == gold_chunk["doc"])

    hit = any(p["is_gold"] for p in deduped)
    gold_id = corpus.gold_chunk(row["id"])
    gold_entry = corpus.gold_map.get(row["id"]) or {}
    record = {
        "id": row["id"], "set": row.get("set", ""), "pair_id": row.get("pair_id", ""),
        "question": row["question"],
        "gold": [row["doc"], row["node_id"]], "edit_term": row.get("edit_term", ""),
        "gold_chunk_id": gold_id,
        "gold_target_leaf": gold_entry.get("target_leaf"),
        "gold_section_level": gold_entry.get("section_level"),
        "gold_retrieved": hit,
        "gold_rank": next((i + 1 for i, p in enumerate(deduped) if p["is_gold"]), None),
        "n_same_document": sum(1 for p in deduped if p["same_document"]),
        "n_returned": len(deduped),
        "n_unknown_ids": sum(1 for p in deduped if not p["known_id"]),
        "n_quotes_not_verbatim": sum(1 for p in deduped if p["quote_verbatim"] is False),
        "passages": deduped,
        "usage": usage,
        "tokens": usage_tokens(usage),
        "cost": cost_usd(cfg, usage),
        "latency_ms": elapsed_ms,
        "cache_variant": variant,
        "provider": cfg.get("provider_name"),
        "status": response.get("status"),
        "incomplete_reason": (response.get("incomplete_details") or {}).get("reason"),
        "model_id": cfg["model_id"],
        "prefix_sha256": prefix_sha,
        "corpus_sha256": corpus.sha256,
        "parse_error": parse_error,
        "raw_text": text,
        "ts": time.time(),
    }
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw" / f"{row['id']}.json").write_text(
        json.dumps({"response": response, "record": record}, indent=1, default=str),
        encoding="utf-8")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--test", metavar="ID", default=None,
                    help="run exactly one query by id and stop")
    ap.add_argument("--all", action="store_true", help="run every query, resumable")
    ap.add_argument("--queries", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model-id", default=None, help="override config.json")
    ap.add_argument("--provider", default=None, choices=("bedrock", "openai"),
                    help="which endpoint to call; default from config.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-interval", type=float, default=None,
                    help="seconds between call starts; paces under the TPM ceiling")
    args = ap.parse_args(argv)

    if args.provider:
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        raw["provider"] = args.provider
        tmp = Path(args.config).with_suffix(".active.json")
        tmp.write_text(json.dumps(raw))
        cfg = load_config(tmp)
        tmp.unlink(missing_ok=True)
    else:
        cfg = load_config(Path(args.config))
    if args.model_id:
        cfg["model_id"] = args.model_id
    queries_csv = Path(args.queries or (ROOT / cfg["queries_csv"]))
    out_dir = Path(args.out_dir or (ROOT / cfg["results_dir"]))
    rows = list(csv.DictReader(queries_csv.open(encoding="utf-8")))

    _body, prefix = build_request(cfg, rows[0]["question"] if rows else "x")
    print(f"corpus + instructions prefix: {len(prefix):,} chars "
          f"(~{len(prefix) // 4:,}-{int(len(prefix) / 3.4):,} tokens, "
          f"sha256 {hashlib.sha256(prefix.encode()).hexdigest()[:16]})")
    print(f"provider: {cfg.get('provider_name')} | model: {cfg.get('model_id')} | "
          f"endpoint: {cfg.get('base_url')}")
    print(f"cache point: {'yes' if cfg.get('prompt_cache', True) else 'no'}; "
          f"queries: {len(rows)} from {queries_csv.name}")

    if not args.test and not args.all:
        print("\ndry run — nothing sent. Pass --test <ID> for a single call, "
              "or --all for the full set.")
        return 0
    if not cfg.get("model_id"):
        print("\nconfig.json has no model_id. Run discover_models.py with the "
              "credential in the environment, then set it.", file=sys.stderr)
        return 1
    env_name = cfg.get("auth_env", "AWS_BEARER_TOKEN_BEDROCK")
    if not os.environ.get(env_name):
        print(f"\n{env_name} is not set (provider: {cfg.get('provider_name', '?')}).",
              file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    gold_map_path = HERE / "gold_map.json"
    if not gold_map_path.exists():
        print("gold_map.json is missing — run build_gold_map.py first.", file=sys.stderr)
        return 1
    corpus = Corpus(HERE / "corpus_manifest.json", ROOT / "index", gold_map_path)

    if args.test:
        todo = [r for r in rows if r["id"] == args.test]
        if not todo:
            print(f"no query with id {args.test!r}", file=sys.stderr)
            return 1
    else:
        results_path = out_dir / "results.jsonl"
        done = set()
        if results_path.exists():
            # Only completed calls count as done. An errored record must be
            # retried, not skipped, or a transient failure silently removes a
            # query from the eval set forever.
            for line in results_path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if not rec.get("error"):
                    done.add(rec["id"])
        todo = [r for r in rows if r["id"] not in done]
        if args.limit:
            todo = todo[:args.limit]

    results_path = out_dir / ("test.jsonl" if args.test else "results.jsonl")
    interval = float(args.min_interval or cfg.get("min_interval_s") or 0)
    if interval and len(todo) > 1:
        print(f"pacing: {interval:.0f}s between calls "
              f"(~{interval * len(todo) / 60:.0f} min for {len(todo)} queries)")
    last_started = 0.0
    for i, row in enumerate(todo, start=1):
        gap = interval - (time.time() - last_started)
        if last_started and gap > 0:
            print(f"  (pacing {gap:.0f}s)", flush=True)
            time.sleep(gap)
        last_started = time.time()
        print(f"\n[{i}/{len(todo)}] {row['id']} — {row['question'][:90]}")
        try:
            rec = run_one(cfg, corpus, row, out_dir)
        except Exception as exc:  # noqa: BLE001 — record and continue
            print(f"  FAILED: {exc}", file=sys.stderr)
            rec = {"id": row["id"], "error": str(exc), "ts": time.time()}
        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        if "error" in rec:
            continue
        t = rec["tokens"]
        print(f"  returned {rec['n_returned']} passage(s) | gold {rec['gold_chunk_id']} "
              f"({'section' if rec['gold_section_level'] else 'leaf'}) "
              f"retrieved={rec['gold_retrieved']} rank={rec['gold_rank']}")
        print(f"  tokens: input {t['input_tokens']:,} "
              f"(cached {t['cached_tokens']:,}, written {t['cache_write_tokens']:,}, "
              f"fresh {t['fresh_tokens']}) | output {t['output_tokens']} "
              f"(reasoning {t['reasoning_tokens']})")
        cost = rec.get("cost") or {}
        print(f"  {rec['latency_ms'] / 1000:.1f}s | {rec['cache_variant']} | "
              f"status {rec['status']} | ${cost.get('total_usd', 0):.4f} "
              f"[{cost.get('tier', '?')}]")
        if t["cached_tokens"] == 0 and t["cache_write_tokens"] == 0:
            print("  WARNING: no cache read and no cache write — caching is not engaging")
        if rec["n_unknown_ids"]:
            print(f"  WARNING: {rec['n_unknown_ids']} returned id(s) are not in the corpus")
        if rec["n_quotes_not_verbatim"]:
            print(f"  WARNING: {rec['n_quotes_not_verbatim']} quote(s) are not verbatim")
        if rec["parse_error"]:
            print(f"  WARNING: {rec['parse_error']}")
    print(f"\nwrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
