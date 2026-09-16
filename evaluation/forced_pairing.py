"""
Set C — the forced-pairing control.

Hands each Set A question a passage set that cannot answer it, skips
retrieval entirely, and calls the product's synthesis step directly through
synthesis.synthesise — the same function, prompt and parsing the chat
endpoint uses. The model should say the evidence is insufficient every time.
This is a floor check, not a headline metric.

Three ways to build the irrelevant passage set:

    --source derange --from-run evaluation/results/main
        each question gets the passages the SYSTEM retrieved for a different
        question (a cyclic shift over the run's raw payloads). Realistic
        passages, wrong question. Preferred.

    --source gold [--k 3]
        each question gets other questions' gold chunks, straight from the
        index. Needs no prior run.

    --source document --document evaluation/queries/nonmedical_control.md
        each question gets --k consecutive sections of a non-medical document.

    python3 -m evaluation.forced_pairing --queries evaluation/queries/queries.csv \
        --out evaluation/results/forced --source derange --from-run evaluation/results/main

Runs in-process against Ollama (no server needed) — start Ollama first.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from evaluation.common import (Query, append_jsonl, load_corpus_leaves, load_queries,
                               read_jsonl)
import synthesis


def _sources_from_leaves(leaves) -> list[dict]:
    return synthesis.number_sources([{
        "doc": l.doc, "node_id": l.node_id, "title": l.title,
        "breadcrumb": l.breadcrumb, "excerpt": l.content,
    } for l in leaves])


def _document_chunks(path: Path) -> list[dict]:
    """Split a markdown file at its headings into passage dicts."""
    text = Path(path).read_text(encoding="utf-8")
    chunks: list[dict] = []
    heading = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    positions = [(m.start(), m.end(), m.group(2).strip()) for m in heading.finditer(text)]
    for i, (start, end, title) in enumerate(positions):
        stop = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[end:stop].strip()
        if body:
            chunks.append({"doc": path.stem, "node_id": f"section-{i + 1}",
                           "title": title, "breadcrumb": title, "excerpt": body})
    if not chunks:
        chunks.append({"doc": path.stem, "node_id": "whole", "title": path.stem,
                       "breadcrumb": path.stem, "excerpt": text.strip()})
    return chunks


def build_assignments(queries: list[Query], args) -> list[tuple[Query, list[dict], str]]:
    """(query, numbered sources, donor description) for every query."""
    out: list[tuple[Query, list[dict], str]] = []
    n = len(queries)

    if args.source == "derange":
        if not args.from_run:
            raise SystemExit("--source derange needs --from-run <run dir>")
        raw_dir = Path(args.from_run) / "raw"
        pool: list[tuple[Query, list[dict]]] = []
        for q in queries:
            path = raw_dir / f"{q.id}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            sources = (payload.get("grounding") or {}).get("sources") or []
            if sources:
                pool.append((q, sources))
        if len(pool) < 2:
            raise SystemExit("fewer than two queries in the run retrieved anything — "
                             "nothing to derange")
        m = len(pool)
        for i, (q, _) in enumerate(pool):
            # Walk forward until the donor is a different query whose set
            # does not happen to contain this query's own gold chunk.
            for step in range(1, m):
                donor_q, donor_sources = pool[(i + step) % m]
                if all([s["doc"], s["node_id"]] != [q.doc, q.node_id] for s in donor_sources):
                    break
            else:
                donor_q, donor_sources = pool[(i + 1) % m]
            sources = synthesis.number_sources([
                {k: s.get(k) for k in ("doc", "node_id", "title", "breadcrumb", "excerpt")}
                for s in donor_sources])
            out.append((q, sources, f"retrieved set of {donor_q.id}"))
        skipped = n - m
        if skipped:
            print(f"{skipped} queries had no raw payload or retrieved nothing; skipped",
                  file=sys.stderr)

    elif args.source == "gold":
        leaves = {l.key: l for l in load_corpus_leaves()}
        for i, q in enumerate(queries):
            donors = []
            for step in range(1, n):
                other = queries[(i + step) % n]
                if other.gold != q.gold and other.gold in leaves and other.gold not in donors:
                    donors.append(other.gold)
                if len(donors) >= args.k:
                    break
            if not donors:
                print(f"{q.id}: no donor gold chunks available; skipped", file=sys.stderr)
                continue
            out.append((q, _sources_from_leaves([leaves[d] for d in donors]),
                        "gold chunks of " + ", ".join(f"{d}/{nid}" for d, nid in donors)))

    elif args.source == "document":
        if not args.document:
            raise SystemExit("--source document needs --document <markdown file>")
        chunks = _document_chunks(Path(args.document))
        for i, q in enumerate(queries):
            start = (i * args.k) % len(chunks)
            picked = [chunks[(start + j) % len(chunks)] for j in range(min(args.k, len(chunks)))]
            out.append((q, synthesis.number_sources(picked),
                        f"{Path(args.document).name} sections "
                        + ",".join(c["node_id"] for c in picked)))
    else:
        raise SystemExit(f"unknown --source {args.source}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="A")
    ap.add_argument("--source", choices=("derange", "gold", "document"), default="derange")
    ap.add_argument("--from-run", default=None)
    ap.add_argument("--document", default=None)
    ap.add_argument("--k", type=int, default=3, help="passages per query for gold/document")
    ap.add_argument("--model", default=None, help="synthesis model (default: runtime config)")
    ap.add_argument("--ollama-url", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    queries = load_queries(Path(args.queries), sets=args.sets.split(","))
    assignments = build_assignments(queries, args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out / "forced.jsonl")}
    todo = [a for a in assignments if a[0].id not in done]
    if args.limit:
        todo = todo[:args.limit]

    if args.dry_run:
        for q, sources, donor in todo:
            print(f"{q.id}\t{len(sources)} passages\t{donor}")
        print(f"{len(todo)} would run")
        return 0

    (out / "manifest.json").write_text(json.dumps({
        "source": args.source, "from_run": args.from_run, "document": args.document,
        "k": args.k, "queries_file": args.queries, "sets": args.sets,
        "model": args.model, "started": time.time(),
        "prompt": synthesis.build_synthesis_prompt("{query}", "{passages}", completeness=True),
    }, indent=2))

    for i, (q, sources, donor) in enumerate(todo, start=1):
        record = {"id": q.id, "set": q.set, "query": q.question,
                  "gold": [q.doc, q.node_id], "donor": donor,
                  "sources": [[s["doc"], s["node_id"]] for s in sources],
                  "n_sources": len(sources), "expected": "insufficient"}
        try:
            result = synthesis.synthesise(q.question, sources, model=args.model,
                                          url=args.ollama_url, completeness=True)
            status, cited = synthesis.grounding_status(result.answer_text, len(sources))
            record.update({
                "judgment": result.judgment,
                "judgment_label": result.judgment["label"],
                "grounding_status": status,
                "n_cited": len(cited),
                "answer_text": result.answer_text,
                "sections": result.sections,
                "synthesis_ms": result.ms,
                "model": result.model,
                "passed": result.judgment["label"] == "insufficient",
            })
        except Exception as exc:  # noqa: BLE001 — record and continue
            record.update({"error": str(exc), "judgment_label": "error", "passed": False})
        append_jsonl(out / "forced.jsonl", record)
        print(f"[{i}/{len(todo)}] {q.id} label={record.get('judgment_label'):<12} "
              f"grounding={record.get('grounding_status', '-'):<20} "
              f"cited={record.get('n_cited', '-')} {record.get('error', '')}",
              file=sys.stderr, flush=True)

    records = read_jsonl(out / "forced.jsonl")
    passed = sum(1 for r in records if r.get("passed"))
    print(f"{passed}/{len(records)} said insufficient "
          f"({len(records) - passed} did not — see forced.jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
