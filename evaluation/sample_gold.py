"""
Draw the gold chunks for Set A.

A plain random sample of N leaves from the index — no stratification, by
design. Writes a query-file skeleton with `question` and `edit_term` blank,
plus a companion markdown file showing each sampled chunk's text so the
questions can be written against it. A Set B row is emitted next to every
Set A row, sharing its gold chunk, ready for the paired edit.

    python3 -m evaluation.sample_gold --n 100 --seed 1 \
        --out evaluation/queries/queries.csv

The seed is recorded in the sidecar so the sample is reproducible; the
sidecar is where the chunk text lives, so the CSV stays small.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from evaluation.common import Query, load_corpus_leaves, write_queries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--out", default="evaluation/queries/queries.csv")
    ap.add_argument("--min-chars", type=int, default=80,
                    help="skip leaves with less content than this (headings with no prose)")
    args = ap.parse_args(argv)

    leaves = load_corpus_leaves(Path(args.index_dir) if args.index_dir else None)
    eligible = [l for l in leaves if len(l.content.strip()) >= args.min_chars]
    if not eligible:
        print("no eligible leaves in the index")
        return 1
    n = min(args.n, len(eligible))
    if n < args.n:
        print(f"only {n} eligible leaves (asked for {args.n}); sampling all of them")
    rng = random.Random(args.seed)
    sample = rng.sample(eligible, n)

    queries: list[Query] = []
    for i, leaf in enumerate(sample, start=1):
        a_id, b_id = f"A{i:03d}", f"B{i:03d}"
        queries.append(Query(id=a_id, set="A", pair_id="", doc=leaf.doc,
                             node_id=leaf.node_id, question=""))
        queries.append(Query(id=b_id, set="B", pair_id=a_id, doc=leaf.doc,
                             node_id=leaf.node_id, question=""))

    out = Path(args.out)
    write_queries(out, queries)

    sidecar = out.with_suffix(".chunks.md")
    lines = [f"# Sampled gold chunks (n={n}, seed={args.seed}, "
             f"eligible={len(eligible)} of {len(leaves)} leaves)\n"]
    for i, leaf in enumerate(sample, start=1):
        lines.append(f"\n## A{i:03d} / B{i:03d} — {leaf.doc} / {leaf.node_id}\n")
        lines.append(f"_{leaf.breadcrumb}_\n")
        lines.append(leaf.content.strip() + "\n")
    sidecar.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {len(queries)} rows ({n} pairs) to {out}")
    print(f"chunk text for question writing: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
