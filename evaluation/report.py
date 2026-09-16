"""
Aggregate a run directory into the evaluation report.

    python3 -m evaluation.report --run evaluation/results/main \
        [--raised evaluation/results/raised] [--bm25 evaluation/results/bm25] \
        [--frontier evaluation/results/frontier] [--forced evaluation/results/forced]

Writes report.md and report.json into the --run directory. Every number is
computed here from records.jsonl and the baseline files; nothing is typed in.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Optional

from evaluation.common import (CENSORED, GOLD_OUTCOMES, load_records, pair_drift,
                               percentile, read_jsonl)


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({(n / d):.0%})" if d else f"{n}/0"


def _split(records: list[dict]) -> dict[str, list[dict]]:
    return {s: [r for r in records if r["set"] == s] for s in ("A", "B")}


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def judgment_section(records: list[dict]) -> tuple[str, dict]:
    ok = [r for r in records if not r.get("error")]
    raw = Counter((r.get("judgment_raw") or "<missing>").strip()[:60] for r in ok)
    labels = Counter(r["judgment_label"] for r in ok)
    disagree = [r["id"] for r in ok
                if r["judgment_label"] in ("sufficient", "insufficient")
                and r["judgment_label"] != r["judgment_ui_reading"]]
    silent = [r["id"] for r in ok if r["judgment_label"] in ("missing", "unparsed")
              and r["judgment_ui_reading"] == "sufficient"]
    lines = ["## The model's sufficiency label, as emitted", "",
             "Raw `EVIDENCE_SUFFICIENT` strings (top 15), before any interpretation:", ""]
    lines.append("| raw label | count |")
    lines.append("|---|---|")
    for text, n in raw.most_common(15):
        lines.append(f"| `{text}` | {n} |")
    lines += ["", "| interpreted | count |", "|---|---|"]
    for k in ("sufficient", "insufficient", "unparsed", "missing"):
        lines.append(f"| {k} | {labels.get(k, 0)} |")
    lines += ["",
              f"- Label absent or hedged, which the product's prefix match reads as "
              f"*sufficient*: **{len(silent)}** {silent[:10]}",
              f"- Strict reading disagrees with the product's reading: {len(disagree)} {disagree[:10]}",
              ""]
    return "\n".join(lines), {"raw": dict(raw), "labels": dict(labels),
                              "silently_sufficient": silent, "ui_disagreement": disagree}


def cells_section(records: list[dict], title: str, expect: str) -> tuple[str, dict]:
    ok = [r for r in records if not r.get("error")]
    cells = Counter(r["cell"] for r in ok)
    n_judged = sum(v for k, v in cells.items() if k.startswith(("retrieved_", "missed_"))
                   and not k.endswith("unlabeled"))
    rs, ri = cells.get("retrieved_sufficient", 0), cells.get("retrieved_insufficient", 0)
    ms, mi = cells.get("missed_sufficient", 0), cells.get("missed_insufficient", 0)
    lines = [f"## {title}", "",
             f"Expected label for this set: **{expect}**. Deferred gold (censored) and "
             f"unlabeled answers are outside the 2×2.", "",
             "| | model: sufficient | model: insufficient |", "|---|---|---|",
             f"| gold retrieved | {rs} — correct for A, false alarm for B (see below) | {ri} — false alarm for A, correct for B |",
             f"| gold missed | **{ms} — confidently wrong** | {mi} — failing safe |", "",
             f"- **Gold missed + sufficient: {_pct(ms, n_judged)}** of judged answers. "
             f"This is the clinically meaningful number.",
             f"- Gold deferred (censored, not scored): {cells.get('deferred_sufficient', 0) + cells.get('deferred_insufficient', 0) + cells.get('deferred_unlabeled', 0)}",
             f"- Unlabeled (label missing/hedged): retrieved {cells.get('retrieved_unlabeled', 0)}, "
             f"missed {cells.get('missed_unlabeled', 0)}",
             f"- Errors (no answer at all): {len(records) - len(ok)}", ""]
    if expect == "insufficient":
        b_ok = [r for r in ok if r["judgment_label"] == "insufficient"]
        named = [r for r in b_ok if r.get("names_edit")]
        checkable = [r for r in b_ok if r.get("names_edit") is not None]
        with_needed = [r for r in b_ok if r.get("still_needed")]
        lines += [f"- Of the {len(b_ok)} answers saying insufficient: {len(with_needed)} list "
                  f"something under STILL_NEEDED; {len(named)}/{len(checkable)} name the "
                  f"inserted detail (rows with an edit_term only).", ""]
    return "\n".join(lines), dict(cells)


def recall_section(records: list[dict]) -> tuple[str, dict]:
    ok = [r for r in records if not r.get("error")]
    scored = [r for r in ok if r["gold_outcome"] not in CENSORED]
    retrieved = sum(1 for r in scored if r["gold_retrieved"])
    cited = sum(1 for r in scored if r["gold_cited"])
    cited_given_retrieved = sum(1 for r in scored if r["gold_retrieved"] and r["gold_cited"])
    grounding = Counter(r["grounding_status"] for r in ok)
    mean_n = statistics.mean([r["n_retrieved"] for r in ok]) if ok else 0
    lines = ["## Retrieval recall vs answer recall", "",
             "Two different numbers: whether the gold chunk made it into the retrieved "
             "set, and whether the finished answer cited it. Deferred gold excluded "
             "from both denominators.", "",
             f"- Retrieval recall (gold in retrieved set): **{_pct(retrieved, len(scored))}**",
             f"- Answer recall (gold cited inline): **{_pct(cited, len(scored))}**",
             f"- Cited given retrieved: {_pct(cited_given_retrieved, retrieved)} — the gap is "
             f"the synthesiser dropping a passage it was handed",
             f"- Mean retrieved set size: {mean_n:.1f} passages",
             f"- Mechanical grounding status: {dict(grounding)}", ""]
    return "\n".join(lines), {"retrieval_recall": [retrieved, len(scored)],
                              "answer_recall": [cited, len(scored)],
                              "cited_given_retrieved": [cited_given_retrieved, retrieved],
                              "mean_retrieved": mean_n, "grounding": dict(grounding)}


def _rank_buckets(ranks: list[int]) -> str:
    if not ranks:
        return "n/a"
    r1 = [r + 1 for r in ranks]  # 0-based → 1-based
    return (f"median {statistics.median(r1):.0f}, p25 {percentile(r1, 25):.0f}, "
            f"p75 {percentile(r1, 75):.0f}; ≤5: {sum(1 for r in r1 if r <= 5)}, "
            f"6–20: {sum(1 for r in r1 if 5 < r <= 20)}, >20: {sum(1 for r in r1 if r > 20)}")


def death_section(records: list[dict], bm25_rows: Optional[dict] = None) -> tuple[str, dict]:
    ok = [r for r in records if not r.get("error")]
    outcomes = Counter(r["gold_outcome"] for r in ok)
    lines = ["## Where the gold chunk died", "",
             "Three failures with three different fixes: *pruned* means the tree walk "
             "killed the branch before looking at the leaf; *rejected* means the "
             "evaluator read it and said no; *deferred* means a budget was hit first. "
             "`deferred_judged_relevant` is the one boundary node the evaluator accepted "
             "before the context budget cut it — a correct verdict, not a censored one.", "",
             "| outcome | count |", "|---|---|"]
    for o in GOLD_OUTCOMES:
        if outcomes.get(o):
            lines.append(f"| {o} | {outcomes[o]} |")
    lines += ["", "In-run BM25 rank of the gold chunk (among post-pruning survivors, 1-based):", ""]
    detail = {}
    for o in ("rejected", "deferred", "deferred_judged_relevant", "retrieved"):
        ranks = [r["gold_bm25_rank"] for r in ok if r["gold_outcome"] == o
                 and r.get("gold_bm25_rank") is not None]
        detail[o] = ranks
        lines.append(f"- {o}: {_rank_buckets(ranks)}")
    lines += ["", "Pruned gold has no in-run rank (it was never a candidate). "
              "Rejected gold ranked highly means the evaluator erred; deferred gold "
              "ranked low means the ranking is the bottleneck and the evaluator never "
              "had a chance.", ""]
    if bm25_rows:
        for o in ("pruned", "rejected", "deferred"):
            ranks = [bm25_rows[r["id"]]["gold_rank"] for r in ok
                     if r["gold_outcome"] == o and r["id"] in bm25_rows
                     and bm25_rows[r["id"]].get("gold_rank")]
            if ranks:
                lines.append(f"- corpus-wide BM25 rank of {o} gold: median "
                             f"{statistics.median(ranks):.0f} (n={len(ranks)})")
        lines.append("")
    return "\n".join(lines), {"outcomes": dict(outcomes), "ranks": detail}


def pairs_section(records: list[dict]) -> tuple[str, dict]:
    by_id = {r["id"]: r for r in records if not r.get("error")}
    pairs = [(by_id[r["pair_id"]], r) for r in by_id.values()
             if r["set"] == "B" and r.get("pair_id") in by_id]
    drifts = [(a, b, pair_drift(a, b)) for a, b in pairs]
    same = [(a, b) for a, b, d in drifts if d["same_set"]]
    shifted = [(a, b) for a, b, d in drifts if not d["same_set"]]
    transitions = Counter(f"{a['judgment_label']}→{b['judgment_label']}" for a, b in pairs)

    def b_cells(ps):
        return dict(Counter(b["cell"] for _, b in ps))

    lines = ["## Paired edits: did the edit move retrieval?", "",
             f"- Pairs with both rows recorded: {len(pairs)}",
             f"- Retrieved set unchanged by the edit: **{len(same)}** — the clean comparison",
             f"- Retrieved set shifted: {len(shifted)} — still usable, reported separately",
             f"- Gold retrieved in both rows: {sum(1 for _, _, d in drifts if d['gold_in_both'])}",
             f"- Mean Jaccard of the two retrieved sets: "
             f"{(statistics.mean(d['jaccard'] for _, _, d in drifts) if drifts else 0):.2f}", "",
             "A→B judgment transitions (expected: sufficient→insufficient):", ""]
    for k, n in transitions.most_common():
        lines.append(f"- {k}: {n}")
    lines += ["", f"Set B cells, pairs with unchanged retrieval: {b_cells(same)}",
              f"Set B cells, pairs with shifted retrieval: {b_cells(shifted)}", ""]
    if shifted:
        lines.append("Shifted pairs: " + ", ".join(f"{b['id']}" for _, b in shifted[:30]))
        lines.append("")
    return "\n".join(lines), {"pairs": len(pairs), "same": len(same), "shifted": len(shifted),
                              "shifted_ids": [b["id"] for _, b in shifted],
                              "transitions": dict(transitions),
                              "b_cells_same": b_cells(same), "b_cells_shifted": b_cells(shifted)}


def budget_section(main: list[dict], raised: Optional[list[dict]]) -> tuple[str, dict]:
    ok = [r for r in main if not r.get("error")]
    deferred = [r for r in ok if r["gold_outcome"] in ("deferred", "deferred_judged_relevant")]
    capped = Counter((r.get("budget") or {}).get("capped_by") or "none" for r in ok)
    lines = ["## Budget: main arm vs raised-cap arm", "",
             f"- Runs capped by: {dict(capped)}",
             f"- Gold deferred in the main arm: {len(deferred)} "
             f"(censored — never folded into 'missed')", ""]
    out = {"deferred_main": len(deferred), "capped_by": dict(capped)}
    if raised is None:
        lines += ["No raised-cap arm supplied (`--raised`). Run it only over the queries "
                  "whose gold was deferred: `run_eval --only-gold-deferred-from`.", ""]
    else:
        by_id = {r["id"]: r for r in raised if not r.get("error")}
        rerun = [(r, by_id[r["id"]]) for r in deferred if r["id"] in by_id]
        after = Counter(b["gold_outcome"] for _, b in rerun)
        recovered = after.get("retrieved", 0)
        cfg = {}
        lines += [f"- Re-run with the cap raised: {len(rerun)} of {len(deferred)}",
                  f"- Outcome after raising the cap: {dict(after)}",
                  f"- Recovered into the retrieved set: **{recovered}** — that is the cost "
                  f"of the production budget in recall points "
                  f"({recovered}/{len(ok) - 0} = {(recovered / len(ok)):.1%} of all judged "
                  f"queries)" if ok else "",
                  "",
                  "The raised-cap arm is diagnostic only. config.py puts the model's quality "
                  "band at 32–64k tokens with a 48k default, well below its 128k maximum; a "
                  "wide-open window shows what retrieval could reach, not what should ship.",
                  ""]
        out.update({"rerun": len(rerun), "after": dict(after), "recovered": recovered,
                    "raised_config": cfg})
    return "\n".join(lines), out


def latency_section(records: list[dict]) -> tuple[str, dict]:
    ok = [r for r in records if not r.get("error")]
    phases = ("retrieval_ms", "prune_ms", "evaluate_ms", "synthesis_ms")
    lines = ["## Latency, per phase (wall-clock ms)", "",
             "| set | phase | n | p50 | p95 |", "|---|---|---|---|---|"]
    out = {}
    for label, subset in [("all", ok)] + list(_split(ok).items()):
        for ph in phases:
            vals = [(r.get("timing") or {}).get(ph) for r in subset]
            vals = [v for v in vals if v is not None]
            p50, p95 = percentile(vals, 50), percentile(vals, 95)
            out[f"{label}.{ph}"] = {"n": len(vals), "p50": p50, "p95": p95}
            lines.append(f"| {label} | {ph} | {len(vals)} | "
                         f"{p50 if p50 is None else int(p50)} | "
                         f"{p95 if p95 is None else int(p95)} |")
    lines.append("")
    return "\n".join(lines), out


def baselines_section(main: list[dict], bm25_dir: Optional[Path],
                      frontier_dir: Optional[Path], forced_dir: Optional[Path]) -> tuple[str, dict]:
    lines = ["## Baselines", ""]
    out: dict = {}
    ok = [r for r in main if not r.get("error") and r["gold_outcome"] not in CENSORED]
    sys_recall = sum(1 for r in ok if r["gold_retrieved"]) / len(ok) if ok else None
    lines.append(f"- System retrieval recall (for comparison): "
                 f"{'n/a' if sys_recall is None else f'{sys_recall:.1%}'}")

    if bm25_dir and (bm25_dir / "bm25_summary.json").exists():
        s = json.loads((bm25_dir / "bm25_summary.json").read_text())
        bm = s.get("budget_matched") or {}
        lines.append(f"- **BM25 alone** (lower bound), whole corpus of {s['corpus_leaves']} leaves:")
        for k in s["ks"]:
            v = s["all"].get(f"recall@{k}")
            tag = "  ← budget-matched" if bm and k == bm.get("k") else ""
            lines.append(f"    - recall@{k}: {v:.1%}{tag}")
        lines.append(f"    - MRR {s['all']['mrr']:.3f}, median gold rank {s['all']['median_rank']}")
        out["bm25"] = s
    else:
        lines.append("- BM25 baseline: not supplied (`--bm25`)")

    if frontier_dir and (frontier_dir / "frontier.jsonl").exists():
        rows = [r for r in read_jsonl(frontier_dir / "frontier.jsonl") if not r.get("error")]
        hits = sum(1 for r in rows if r.get("gold_cited"))
        misses = sum(1 for r in rows if r.get("cache") == "MISS")
        mean_cited = statistics.mean([r["n_cited"] for r in rows]) if rows else 0
        man = json.loads((frontier_dir / "manifest.json").read_text()) \
            if (frontier_dir / "manifest.json").exists() else {}
        lines += [f"- **Frontier model, whole corpus in context** (upper bound; "
                  f"{man.get('model', '?')}): gold cited {_pct(hits, len(rows))}, "
                  f"mean cited {mean_cited:.2f}, cache misses {misses}/{len(rows)}",
                  "    - A ceiling, not a target: it cannot be used clinically. Note the "
                  "question-generation leakage direction: questions written from chunk text "
                  "share its vocabulary, inflating BM25 and the system but not this baseline, "
                  "so the gap to the ceiling is understated."]
        out["frontier"] = {"gold_cited": [hits, len(rows)], "mean_cited": mean_cited,
                           "cache_misses": misses}
    else:
        lines.append("- Frontier baseline: not supplied (`--frontier`)")

    if forced_dir and (forced_dir / "forced.jsonl").exists():
        rows = read_jsonl(forced_dir / "forced.jsonl")
        passed = sum(1 for r in rows if r.get("passed"))
        grounded = sum(1 for r in rows if r.get("grounding_status") == "grounded")
        labels = Counter(r.get("judgment_label") for r in rows)
        lines += [f"- **Forced pairing** (Set C floor check): said insufficient "
                  f"{_pct(passed, len(rows))}; labels {dict(labels)}; cited an irrelevant "
                  f"passage anyway in {grounded} answers"]
        out["forced"] = {"passed": [passed, len(rows)], "labels": dict(labels),
                         "grounded": grounded}
    else:
        lines.append("- Forced-pairing control: not supplied (`--forced`)")
    lines.append("")
    return "\n".join(lines), out


LIMITATIONS = """\
## Stated limitations

- **Test–retest variance** is not measured; each query ran once. Accepted as a limitation.
- **Sampling** is a plain random draw of leaves, unstratified, by design.
- **Multi-chunk questions** are deferred; every query has exactly one gold chunk.
- **Ingestion fidelity** is out of scope: the corpus is hand-written markdown, so there is no parser to evaluate.
- **Question-generation leakage**: questions written from the chunk text share its vocabulary. That inflates BM25, and therefore the system built on it, while leaving the full-text frontier baseline unaffected — the bias runs in the system's favour and matters when reporting how close the system came to the ceiling.
- **The raised-cap arm is diagnostic only**: the model's quality band sits well below its maximum context, so it shows what retrieval could reach, not what should ship.
- **Deferred is censored**, not missed: the system never formed a judgment, and it is reported apart from the 2×2 throughout.
"""


def build(run_dir: Path, raised_dir: Optional[Path], bm25_dir: Optional[Path],
          frontier_dir: Optional[Path], forced_dir: Optional[Path]) -> tuple[str, dict]:
    records = load_records(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text()) \
        if (run_dir / "manifest.json").exists() else {}
    raised = load_records(raised_dir) if raised_dir else None
    bm25_rows = {r["id"]: r for r in read_jsonl(bm25_dir / "bm25.jsonl")} \
        if bm25_dir and (bm25_dir / "bm25.jsonl").exists() else None

    fps = Counter(r.get("index_fingerprint") for r in records)
    sets = _split(records)
    cfg = manifest.get("config") or {}
    header = [
        "# Asepsis retrieval evaluation", "",
        f"- Run: `{run_dir}` — arm `{manifest.get('arm', '?')}`, "
        f"{len(records)} records (A={len(sets['A'])}, B={len(sets['B'])}), "
        f"{sum(1 for r in records if r.get('error'))} errors",
        f"- Index fingerprint(s) seen: {dict(fps)}"
        + ("  ⚠ MORE THAN ONE — the corpus changed mid-run" if len(fps) > 1 else ""),
        f"- Aborted: {manifest.get('aborted', False)} {manifest.get('abort_reason', '')}",
        f"- Config: retrieval {cfg.get('retrieval_model')} @ {cfg.get('retrieval_ctx')}, "
        f"synthesis {cfg.get('synthesis_model')} @ agent_ctx {cfg.get('agent_ctx')}, "
        f"max_leaf_evals {cfg.get('max_leaf_evals')}, "
        f"completeness_check {cfg.get('completeness_check')}",
        "",
        "Two independent signals are reported and never conflated: the mechanical "
        "grounding status (no evidence / cited / uncited) needs no evaluation; the "
        "model's own sufficiency judgment is what is evaluated below.", "",
    ]

    parts, data = ["\n".join(header)], {"manifest": manifest, "fingerprints": dict(fps)}
    for name, (text, d) in {
        "judgment": judgment_section(records),
        "cells_A": cells_section(sets["A"], "Set A — gold queries", "sufficient"),
        "cells_B": cells_section(sets["B"], "Set B — paired edits", "insufficient"),
        "recall": recall_section(records),
        "death": death_section(records, bm25_rows),
        "pairs": pairs_section(records),
        "budget": budget_section(records, raised),
        "latency": latency_section(records),
        "baselines": baselines_section(records, bm25_dir, frontier_dir, forced_dir),
    }.items():
        parts.append(text)
        data[name] = d
    parts.append(LIMITATIONS)
    return "\n".join(parts), data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--raised", default=None)
    ap.add_argument("--bm25", default=None)
    ap.add_argument("--frontier", default=None)
    ap.add_argument("--forced", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    text, data = build(run_dir,
                       Path(args.raised) if args.raised else None,
                       Path(args.bm25) if args.bm25 else None,
                       Path(args.frontier) if args.frontier else None,
                       Path(args.forced) if args.forced else None)
    (run_dir / "report.md").write_text(text, encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(data, indent=2, default=str),
                                         encoding="utf-8")
    print(text)
    print(f"\nwritten: {run_dir / 'report.md'}, {run_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
