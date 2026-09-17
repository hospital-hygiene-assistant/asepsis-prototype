"""
Golden labels — mark the chunks that should contain a question's answer.

Reads a question CSV (document, page, question) and writes
knowledge_base/.golden.json: {doc_id: {page: [label, ...]}}.  The ingest
module stamps those labels onto the pins of every chunk covering the page, so
"which chunk holds the answer to question 23" is visible in the document and
in the app rather than living in a spreadsheet.

A label is `golden-N`, N being the question's 1-based row in the CSV, so the
label points back at the exact row a reader can look up.

Documents are matched by filename: a CSV row naming `Foo_2026-09-11.pdf` binds
to the doc whose id ENDS WITH `Foo_2026-09-11`, because ingest prefixes ids
with a content hash.  Rows naming a document that is not ingested are reported
and skipped — silently dropping them would overstate eval coverage.

    python evaluation/golden_from_csv.py "~/Downloads/questions (1).csv"
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "knowledge_base"
GOLDEN_PATH = KB_DIR / ".golden.json"


def doc_ids() -> list[str]:
    return sorted(p.stem for p in KB_DIR.glob("*.md"))


def match_doc(name: str, ids: list[str]) -> str | None:
    stem = Path(name.strip()).stem
    for doc_id in ids:
        if doc_id == stem or doc_id.endswith(stem):
            return doc_id
    return None


def build(csv_path: Path) -> tuple[dict, dict]:
    ids = doc_ids()
    golden: dict[str, dict[str, list[str]]] = {}
    unmatched: dict[str, int] = {}
    rows = [r for r in csv.DictReader(csv_path.open(encoding="utf-8"))
            if (r.get("document") or "").strip()]
    for n, row in enumerate(rows, start=1):
        doc_id = match_doc(row["document"], ids)
        if doc_id is None:
            unmatched[row["document"].strip()] = unmatched.get(
                row["document"].strip(), 0) + 1
            continue
        page = str(int(row["page"]))
        golden.setdefault(doc_id, {}).setdefault(page, []).append(f"golden-{n}")
    return golden, unmatched


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <questions.csv>")
    csv_path = Path(sys.argv[1]).expanduser()
    golden, unmatched = build(csv_path)
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True),
                           encoding="utf-8")
    total = sum(len(v) for pages in golden.values() for v in pages.values())
    print(f"wrote {GOLDEN_PATH.relative_to(ROOT)}")
    for doc_id, pages in sorted(golden.items()):
        labels = sorted(l for v in pages.values() for l in v)
        print(f"  {doc_id[:52]:54} {len(pages):3} page(s), {len(labels)} label(s)")
    print(f"  labelled {total} question(s)")
    if unmatched:
        print("\n  NOT INGESTED — these questions have no document to label:")
        for name, n in sorted(unmatched.items(), key=lambda kv: -kv[1]):
            print(f"    {n:3}  {name}")


if __name__ == "__main__":
    main()
