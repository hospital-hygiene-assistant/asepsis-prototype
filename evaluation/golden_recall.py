"""
Golden recall — does retrieval put a golden-labelled chunk in the retrieved set?

    python3 -m evaluation.golden_recall "~/Downloads/questions (1).csv"
    python3 -m evaluation.golden_recall questions.csv --only golden-21,golden-23

One question, one check: for every question that has golden labels (see
golden_from_csv.py), run the real retriever over its document and ask whether
ANY chunk carrying that question's label came back.  Nothing else is scored.

Deliberately narrower than run_eval.py, which drives the chat endpoint and
keys gold on a single (doc, node_id).  A golden label marks a RECALL SET —
a page usually spans several chunks and a parent section inherits its
children's pages — so "did we get the right one" is not a single-id question
here.  Hitting the retriever directly also means no server and no synthesis.

Reports, per question: hit or miss, and for a miss what actually happened to
each labelled chunk (pruned, rescued, never a candidate), because a miss
caused by pruning and a miss caused by the passage not existing need
different fixes.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLDEN_PATH = ROOT / "knowledge_base" / ".golden.json"


def _labels_by_node(stem: str) -> dict[str, set[str]]:
    """{node_id: {label, ...}} straight from the built index."""
    index = json.loads((ROOT / "index" / f"{stem}.json").read_text())
    out: dict[str, set[str]] = {}

    def walk(nodes):
        for n in nodes:
            golden = (n.get("pin") or {}).get("golden")
            if golden:
                out[n["nodeId"]] = {s.strip() for s in str(golden).split(",")}
            walk(n.get("children") or [])

    walk(index["nodes"])
    return out


def main() -> None:
    from modules.index import pageindex_custom as ix

    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <questions.csv> [--only label,label]")
    only: set[str] = set()
    if "--only" in sys.argv:
        only = {s.strip() for s in sys.argv[sys.argv.index("--only") + 1].split(",")}
    rows = [r for r in csv.DictReader(Path(sys.argv[1]).expanduser().open(encoding="utf-8"))
            if (r.get("document") or "").strip()]
    golden_map = json.loads(GOLDEN_PATH.read_text())

    # label -> (stem, question); only questions we actually labelled
    wanted: dict[str, tuple[str, str]] = {}
    for stem, pages in golden_map.items():
        for labels in pages.values():
            for label in labels:
                n = int(label.split("-")[1])
                wanted[label] = (stem, rows[n - 1]["question"])

    hits = misses = 0
    detail: list[str] = []
    order = sorted((l for l in wanted if not only or l in only),
                   key=lambda s: int(s.split("-")[1]))
    # flush=True on every line: stdout block-buffers when redirected to a file,
    # so without it a run that takes a minute per question shows nothing at all
    # until it finishes — which is exactly when a progress log is useless.
    print(f"checking {len(order)} labelled question(s)", flush=True)
    for i, label in enumerate(order, start=1):
        stem, question = wanted[label]
        print(f"  [{i}/{len(order)}] {label} …", flush=True)
        by_node = _labels_by_node(stem)
        targets = {nid for nid, labs in by_node.items() if label in labs}
        nodes, meta = ix.retrieve_with_metadata(stem, question)
        got = {n.node_id for n in nodes}
        found = sorted(targets & got)
        if found:
            hits += 1
            nid = found[0]
            m = meta.get(nid) or {}
            how = "rehydrated" if m.get("rehydrated") else "walk"
            # A rehydrated leaf is only a hit because the EVALUATOR accepted
            # it; before, the rehydration itself set relevant=True and the
            # hit could be an artefact of that. Print the verdict so the two
            # are never confused again.
            verdict = "evaluator: accepted" if m.get("relevant") else "NOT EVALUATED"
            print(f"  {label:>10}  HIT   {nid[:44]:46} ({how}, {verdict})", flush=True)
        else:
            misses += 1
            print(f"  {label:>10}  MISS  {len(targets)} labelled chunk(s) not retrieved", flush=True)
            for nid in sorted(targets):
                m = meta.get(nid) or {}
                detail.append(f"      {label} {nid[:44]:46} "
                              f"{m.get('status', 'never a candidate')}: "
                              f"{str(m.get('reason', ''))[:80]}")
    total = hits + misses
    print()
    print(f"golden recall: {hits}/{total} ({hits / total:.0%})" if total else "no labels")
    if detail:
        print("\nmisses in detail:")
        print("\n".join(detail))


if __name__ == "__main__":
    main()
