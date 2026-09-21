"""
Write one prompt file per query.

Each file holds exactly what is sent AFTER the cached corpus block: the task
instructions followed by that query's question. The corpus itself is not
copied into the 100 files, for two reasons. It is byte-identical across every
call, and that identity is precisely what lets it be cached; and 100 copies of
a 2 MB corpus is 200 MB of duplicated text that can silently drift out of sync
with the one file the runner actually sends.

    python3 llm-baseline/build_prompts.py                      # Set A
    python3 llm-baseline/build_prompts.py --queries evaluation/queries/asepsis_100_fictional.csv \
        --out-dir llm-baseline/prompts_fictional

Writes <out-dir>/<id>.txt and <out-dir>/index.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

QUESTION_BLOCK = "QUESTION: {question}\n\nReply with the JSON object only.\n"


def build_suffix(instructions: str, question: str) -> str:
    """The per-query half of the prompt: static instructions, then the question.

    The instructions sit here rather than in the cached block only in this
    file, which exists to be read. The runner places them INSIDE the cached
    prefix, since they never vary; see run_baseline.py.
    """
    return f"{instructions.rstrip()}\n\n{'-' * 78}\n\n{QUESTION_BLOCK.format(question=question)}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default=str(ROOT / "evaluation/queries/asepsis_100.csv"))
    ap.add_argument("--instructions", default=str(HERE / "instructions.txt"))
    ap.add_argument("--out-dir", default=str(HERE / "prompts"))
    args = ap.parse_args(argv)

    instructions = Path(args.instructions).read_text(encoding="utf-8")
    rows = [r for r in csv.DictReader(Path(args.queries).open(encoding="utf-8"))
            if (r.get("question") or "").strip()]
    if not rows:
        print(f"no questions in {args.queries}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for row in rows:
        suffix = build_suffix(instructions, row["question"].strip())
        path = out_dir / f"{row['id']}.txt"
        path.write_text(suffix, encoding="utf-8")
        written.append({
            "id": row["id"],
            "set": row.get("set", ""),
            "pair_id": row.get("pair_id", ""),
            "gold_doc": row["doc"],
            "gold_node_id": row["node_id"],
            "edit_term": row.get("edit_term", ""),
            "question": row["question"].strip(),
            "file": path.name,
            "chars": len(suffix),
            "sha256": hashlib.sha256(suffix.encode("utf-8")).hexdigest()[:16],
        })

    (out_dir / "index.json").write_text(json.dumps({
        "queries_csv": str(args.queries),
        "instructions_sha256": hashlib.sha256(
            instructions.encode("utf-8")).hexdigest()[:16],
        "n": len(written),
        "prompts": written,
    }, indent=1), encoding="utf-8")

    sizes = [w["chars"] for w in written]
    print(f"wrote {len(written)} prompt files to {out_dir}")
    print(f"  per-file size {min(sizes):,}-{max(sizes):,} chars "
          f"(instructions {len(instructions):,} + the question)")
    print("  the corpus prefix is NOT duplicated here; the runner prepends "
          "corpus_context.txt at call time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
