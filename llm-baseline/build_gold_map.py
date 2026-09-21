"""
Resolve each question's gold label to the ONE leaf that holds the answer.

The gold labels are carried by the pin blocks in knowledge_base/*.md: a pin
holds `golden: golden-N, golden-M`, where N is the question's 1-based row in
the query CSV. A pin sits immediately before the text it describes, so the
label marks THAT text and nothing else.

Two cases, and the difference matters:

  pin on a leaf      the gold chunk is that leaf.

  pin on an internal section
                     the section holds no text of its own. Its prose is the
                     preamble between its heading and its first subheading,
                     which `_promote_preambles` lifts into a synthetic
                     `<section>-overview` leaf, copying the section's pin onto
                     it (`pin=node.pin`). So the label travels to the overview
                     leaf and to nothing else. The section's other subsections
                     have their own pins and do NOT carry the label.

The gold chunk is therefore the overview leaf, NOT "any leaf beneath the
section". This script proves that rather than assuming it: it checks that no
non-overview descendant of a golden section carries the same label, and fails
if one does.

    python3 llm-baseline/build_gold_map.py

Writes llm-baseline/gold_map.json: {query_id: {doc, gold_node_id, target_leaf,
chunk_id, section_level}}. This is scoring data and never enters a prompt.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pageindex  # noqa: E402

PIN = re.compile(r"^```pin\s*$\n(.*?)^```\s*$\n?", re.M | re.S)
OVERVIEW_SUFFIX = "-overview"


def read_label_pins(kb_dir: Path) -> dict[str, tuple[str, str]]:
    """{golden-N: (doc, pinned_node_id)} straight from the markdown pins."""
    out: dict[str, tuple[str, str]] = {}
    for md in sorted(kb_dir.glob("*.md")):
        for match in PIN.finditer(md.read_text(encoding="utf-8")):
            body = match.group(1)
            golden = re.search(r"^golden:\s*(.+)$", body, re.M)
            node = re.search(r"^id:\s*(\S+)", body, re.M)
            if not (golden and node):
                continue
            for label in (x.strip() for x in golden.group(1).split(",")):
                if not label:
                    continue
                if label in out:
                    raise SystemExit(
                        f"{label} appears on two pins: {out[label]} and "
                        f"{(md.stem, node.group(1))}. A label must mark one chunk.")
                out[label] = (md.stem, node.group(1))
    return out


def load_trees(index_dir: Path) -> dict[str, dict]:
    trees = {}
    for path in sorted(index_dir.glob("*.json")):
        nodes = [pageindex._node_from_dict(d)
                 for d in pageindex.read_index_file(path)["nodes"]]
        trees[path.stem] = {n.node_id: n for n in pageindex._collect_all_nodes(nodes)}
    return trees


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default=str(ROOT / "evaluation/queries/asepsis_100.csv"))
    ap.add_argument("--manifest", default=str(HERE / "corpus_manifest.json"))
    ap.add_argument("--out", default=str(HERE / "gold_map.json"))
    args = ap.parse_args(argv)

    label_pins = read_label_pins(ROOT / "knowledge_base")
    trees = load_trees(ROOT / "index")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    chunk_of = {(c["doc"], c["node_id"]): c["chunk_id"] for c in manifest["chunks"]}
    rows = list(csv.DictReader(Path(args.queries).open(encoding="utf-8")))

    out: dict[str, dict] = {}
    problems: list[str] = []
    n_leaf = n_section = 0

    for i, row in enumerate(rows, start=1):
        label = f"golden-{i}"
        pinned = label_pins.get(label)
        if pinned is None:
            problems.append(f"{row['id']}: no pin carries {label}")
            continue
        doc, node_id = pinned
        # The CSV must agree with the pins, or the row numbering has drifted
        # and every label after the drift points at the wrong chunk.
        if doc != row["doc"] or node_id != row["node_id"]:
            problems.append(f"{row['id']}: pin says {doc}/{node_id}, CSV says "
                            f"{row['doc']}/{row['node_id']}")
            continue
        node = trees.get(doc, {}).get(node_id)
        if node is None:
            problems.append(f"{row['id']}: {doc}/{node_id} is not in the index")
            continue

        if node.is_leaf:
            target = node_id
            n_leaf += 1
        else:
            target = node_id + OVERVIEW_SUFFIX
            n_section += 1
            if not any(c.node_id == target for c in node.children):
                problems.append(f"{row['id']}: section {node_id} has no {target} leaf")
                continue
            # The claim under test: the label marks the overview and nothing else.
            strays = [leaf.node_id for leaf in pageindex._collect_leaves([node])
                      if leaf.node_id != target
                      and label in _labels_on(leaf, doc, label_pins)]
            if strays:
                problems.append(f"{row['id']}: {label} also marks non-overview "
                                f"descendants {strays}")
                continue

        chunk_id = chunk_of.get((doc, target))
        if chunk_id is None:
            problems.append(f"{row['id']}: target leaf {doc}/{target} is not a corpus chunk")
            continue
        out[row["id"]] = {"doc": doc, "gold_node_id": node_id, "target_leaf": target,
                          "chunk_id": chunk_id, "section_level": not node.is_leaf,
                          "label": label}

    if problems:
        print("gold map NOT written — unresolved labels:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  {len(out)} queries resolved to exactly one leaf each")
    print(f"  gold pin on a leaf (target is itself)        : {n_leaf}")
    print(f"  gold pin on a section (target is -overview)  : {n_section}")
    print("  verified: no non-overview descendant carries a section's label")
    return 0


def _labels_on(leaf, doc: str, label_pins: dict) -> set[str]:
    return {lab for lab, (d, nid) in label_pins.items()
            if d == doc and nid == leaf.node_id}


if __name__ == "__main__":
    raise SystemExit(main())
