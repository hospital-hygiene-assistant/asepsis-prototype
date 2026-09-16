"""
Lower bound: BM25 alone, over every leaf in the corpus. No LLM calls.

The system ranks its post-pruning candidates with this same BM25 (ranking.py,
same tokeniser, same title+summary+content text) and then spends one model
call per leaf accepting or rejecting in that order. So the question this
answers is whether the LLM layer improves on the ranking it is built on.

Two ranks exist and both are useful:
  - the rank here, among EVERY leaf in the corpus
  - the bm25_rank recorded inside a run, among post-pruning survivors only
They are different numbers. The report joins them per query.

    python3 -m evaluation.bm25_baseline --queries evaluation/queries/queries.csv \
        --out evaluation/results/bm25 --budget-from evaluation/results/main

--budget-from sets k to the mean number of passages the system actually
retrieved in that run (budget-matched recall@k), alongside fixed ks.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from evaluation.common import (append_jsonl, load_corpus_leaves, load_queries,
                               load_records, rank_corpus)

DEFAULT_KS = (1, 3, 5, 10, 20, 50)


def budget_matched_k(run_dir: Path) -> dict:
    records = [r for r in load_records(run_dir) if not r.get("error")]
    counts = [r.get("n_retrieved", 0) for r in records]
    if not counts:
        return {}
    mean = statistics.mean(counts)
    return {"mean_retrieved": mean, "median_retrieved": statistics.median(counts),
            "k": max(1, int(round(mean))), "n_runs": len(counts)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="A,B")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--budget-from", default=None,
                    help="run dir whose mean retrieved count sets the budget-matched k")
    args = ap.parse_args(argv)

    leaves = load_corpus_leaves(Path(args.index_dir) if args.index_dir else None)
    if not leaves:
        print("no leaves in the index", file=sys.stderr)
        return 1
    queries = load_queries(Path(args.queries), sets=args.sets.split(","))
    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})
    budget = budget_matched_k(Path(args.budget_from)) if args.budget_from else {}
    if budget:
        ks = sorted(set(ks) | {budget["k"]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_path = out / "bm25.jsonl"
    per_path.unlink(missing_ok=True)

    rows = []
    for q in queries:
        ranked = rank_corpus(leaves, q.question)
        position = next((s.rank for s in ranked if s.item.key == q.gold), None)
        top = [s.item.chunk_id for s in ranked[:max(ks)]]
        row = {
            "id": q.id, "set": q.set, "gold": [q.doc, q.node_id],
            "gold_rank": (position + 1) if position is not None else None,  # 1-based
            "gold_score": (ranked[position].score if position is not None else None),
            "gold_in_index": position is not None,
            "top": top[:20],
            **{f"recall@{k}": (position is not None and position < k) for k in ks},
        }
        rows.append(row)
        append_jsonl(per_path, row)

    def summarise(subset: list[dict]) -> dict:
        n = len(subset)
        ranks = [r["gold_rank"] for r in subset if r["gold_rank"]]
        return {
            "n": n,
            **{f"recall@{k}": (sum(1 for r in subset if r[f"recall@{k}"]) / n if n else None)
               for k in ks},
            "mrr": (sum(1.0 / r for r in ranks) / n) if n else None,
            "median_rank": statistics.median(ranks) if ranks else None,
            "gold_missing_from_index": sum(1 for r in subset if not r["gold_in_index"]),
        }

    summary = {
        "corpus_leaves": len(leaves),
        "ks": ks,
        "budget_matched": budget,
        "all": summarise(rows),
        "by_set": {s: summarise([r for r in rows if r["set"] == s])
                   for s in sorted({r["set"] for r in rows})},
    }
    (out / "bm25_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"corpus: {len(leaves)} leaves; {len(rows)} queries")
    if budget:
        print(f"budget-matched k = {budget['k']} (mean retrieved {budget['mean_retrieved']:.2f} "
              f"over {budget['n_runs']} runs)")
    header = "set   n   " + "  ".join(f"R@{k:<3}" for k in ks) + "  MRR    median"
    print(header)
    for label, s in [("all", summary["all"])] + list(summary["by_set"].items()):
        cells = "  ".join(f"{(s[f'recall@{k}'] or 0):.2f} " for k in ks)
        print(f"{label:<4} {s['n']:>3}  {cells}  {(s['mrr'] or 0):.3f}  {s['median_rank']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
