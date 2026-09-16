"""
Check a query file before spending an afternoon of model calls on it.

    python3 -m evaluation.validate_queries evaluation/queries/queries.csv

Fails on: a gold chunk that is not a leaf in the index, a duplicate id, a B
row without an A partner, a pair whose two rows name different gold chunks,
or a blank question. Warns on a B row without an edit_term (the report can
then not check whether STILL_NEEDED names the edit).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.common import load_corpus_leaves, load_queries


def validate(path: Path, index_dir: Path | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    queries = load_queries(path)
    leaves = {l.key for l in load_corpus_leaves(index_dir)}

    seen: set[str] = set()
    by_id = {}
    for q in queries:
        if q.id in seen:
            errors.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        by_id[q.id] = q
        if q.set not in ("A", "B"):
            errors.append(f"{q.id}: set must be A or B, got {q.set!r}")
        if not q.question:
            errors.append(f"{q.id}: blank question")
        if q.gold not in leaves:
            errors.append(f"{q.id}: gold {q.doc}/{q.node_id} is not a leaf in the index")

    for q in queries:
        if q.set != "B":
            continue
        if not q.pair_id:
            errors.append(f"{q.id}: B row has no pair_id")
            continue
        a = by_id.get(q.pair_id)
        if a is None or a.set != "A":
            errors.append(f"{q.id}: pair_id {q.pair_id!r} is not a Set A row")
            continue
        if a.gold != q.gold:
            errors.append(f"{q.id}: gold differs from its pair {a.id} "
                          f"({q.doc}/{q.node_id} vs {a.doc}/{a.node_id})")
        if a.question and q.question and a.question == q.question:
            errors.append(f"{q.id}: identical to its pair {a.id} — no edit was made")
        if not q.edit_term:
            warnings.append(f"{q.id}: no edit_term; the report cannot check whether "
                            "STILL_NEEDED names the inserted detail")
    return errors, warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queries")
    ap.add_argument("--index-dir", default=None)
    args = ap.parse_args(argv)
    errors, warnings = validate(Path(args.queries),
                                Path(args.index_dir) if args.index_dir else None)
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"error: {e}")
    n = len(load_queries(Path(args.queries)))
    print(f"{n} rows, {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
