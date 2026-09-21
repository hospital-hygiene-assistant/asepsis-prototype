"""
Estimate what a full baseline run costs, before spending anything.

Token counts come from the o200k_base encoding, which is an estimate: the
exact count is whatever Bedrock reports in `usage`, and the first real call
replaces this with a measured figure. Prices come from config.json, which
carries the GPT-5.6 Luna model card's two tiers.

The tier boundary is the thing to notice. Bedrock charges double above 272K
input tokens, and this corpus is well past that, so every call is billed at
the long-context rate.

    python3 llm-baseline/estimate_cost.py
    python3 llm-baseline/estimate_cost.py --queries 200   # Set A + Set B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def prefix_tokens() -> tuple[int, str]:
    corpus = (HERE / "corpus_context.txt").read_text(encoding="utf-8")
    instructions = (HERE / "instructions.txt").read_text(encoding="utf-8")
    text = f"{corpus.rstrip()}\n\n{'=' * 78}\n\n{instructions.rstrip()}"
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(text)), "o200k_base"
    except Exception:
        return len(text) // 4, "4 chars/token fallback"


def scenario(p: dict, n: int, prefix: int, out_tokens: int, suffix: int,
             cached: bool) -> dict:
    if cached:
        write = prefix * p["cache_write"] / 1e6
        read = (n - 1) * prefix * p["cache_read"] / 1e6
        fresh = n * suffix * p["input"] / 1e6
    else:
        write = read = 0.0
        fresh = n * (prefix + suffix) * p["input"] / 1e6
    out = n * out_tokens * p["output"] / 1e6
    return {"cache_write": write, "cache_read": read, "fresh_input": fresh,
            "output": out, "total": write + read + fresh + out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--suffix-tokens", type=int, default=40)
    ap.add_argument("--out-low", type=int, default=500)
    ap.add_argument("--out-mid", type=int, default=1500)
    ap.add_argument("--out-high", type=int, default=4000)
    ap.add_argument("--provider", default=None, choices=("bedrock", "openai"))
    args = ap.parse_args(argv)

    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    name = args.provider or cfg.get("provider")
    prov = (cfg.get("providers") or {}).get(name) or {}
    if prov.get("pricing_usd_per_mtok"):
        cfg["pricing_usd_per_mtok"] = prov["pricing_usd_per_mtok"]
        cfg.pop("pricing_usd_per_mtok_global_profile", None)
    print(f"provider: {name}  model: {prov.get('model_id')}")
    n = args.queries
    prefix, how = prefix_tokens()
    threshold = cfg.get("long_context_threshold_tokens", 272000)
    tier = "long" if prefix > threshold else "short"

    print(f"prefix: {prefix:,} tokens ({how})")
    print(f"tier  : {tier}  (threshold {threshold:,}; above it every rate doubles)")
    print(f"calls : {n}\n")

    for label, key in ((f"{name} list price", "pricing_usd_per_mtok"),
                       ("global. profile", "pricing_usd_per_mtok_global_profile")):
        table = cfg.get(key)
        if not table:
            continue
        p = table[tier]
        print(f"=== {label} — ${p['input']}/M in, ${p['cache_write']}/M write, "
              f"${p['cache_read']}/M read, ${p['output']}/M out")
        for cached in (True, False):
            head = "with prompt caching" if cached else "WITHOUT caching"
            lo = scenario(p, n, prefix, args.out_low, args.suffix_tokens, cached)
            mid = scenario(p, n, prefix, args.out_mid, args.suffix_tokens, cached)
            hi = scenario(p, n, prefix, args.out_high, args.suffix_tokens, cached)
            print(f"  {head:20} ${mid['total']:7.2f}   "
                  f"(range ${lo['total']:.2f}-${hi['total']:.2f} on output volume)")
            print(f"    {'cache write':<14} ${mid['cache_write']:7.2f}   "
                  f"{'cache read':<12} ${mid['cache_read']:7.2f}")
            print(f"    {'fresh input':<14} ${mid['fresh_input']:7.2f}   "
                  f"{'output':<12} ${mid['output']:7.2f}")
        print()

    p = cfg["pricing_usd_per_mtok"][tier]
    saved = (scenario(p, n, prefix, args.out_mid, args.suffix_tokens, False)["total"]
             - scenario(p, n, prefix, args.out_mid, args.suffix_tokens, True)["total"])
    print(f"caching is worth about ${saved:.2f} over {n} calls on the us. profile")
    print("output volume is the one real unknown: reasoning tokens at effort=high are "
          "billed as output and are not visible until the first call returns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
