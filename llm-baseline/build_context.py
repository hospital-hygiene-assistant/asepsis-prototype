"""
Build the cached corpus prefix for the whole-corpus LLM retrieval baseline.

The baseline gives a frontier model the entire knowledge base in one context
and asks it to retrieve the passages relevant to a question. For that to be a
comparison of retrieval against retrieval, it must see the SAME chunks the
system retrieves: the leaves of the heading tree, in the same order the server
walks them (index files sorted by name, then tree order).

Two things must not reach the model.

  1. Pin blocks. They are fenced ```pin ... ``` YAML carrying page, bbox and
     asset provenance — and, on some chunks, a `golden: golden-N` line. That
     line IS the gold label. A corpus pasted straight from knowledge_base/*.md
     would hand the model the answer key. Building from the index sidesteps it:
     pageindex strips pins when it extracts leaf content, so `leaf.content` is
     already pin-free and image-free. This script asserts that rather than
     assuming it.

  2. The chunk-to-node mapping. Chunks are labelled with neutral sequential
     ids (C0001, C0002, ...) assigned in corpus order. The mapping back to
     (doc, node_id) is the scoring key and is written to a SEPARATE manifest
     file that never enters a prompt.

Each chunk carries its breadcrumb, which is derived from the document's own
headings and is what the system's own leaf evaluator sees as "Section path".
LLM-written node summaries are deliberately excluded: they are an artefact of
the system under test, and feeding them to the baseline would make the
baseline depend on the system's own preprocessing.

    python3 llm-baseline/build_context.py

Writes llm-baseline/corpus_context.txt and llm-baseline/corpus_manifest.json.
The prefix is byte-stable: nothing in it carries a timestamp, a per-query id
or anything else that varies between calls, so it can be cached across all
100 calls. Its SHA-256 is recorded in the manifest and checked at run time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pageindex  # noqa: E402

OUT_DIR = ROOT / "llm-baseline"
CONTEXT_PATH = OUT_DIR / "corpus_context.txt"
MANIFEST_PATH = OUT_DIR / "corpus_manifest.json"

# Anything matching these must never appear in the assembled corpus. The pin
# fence and the golden key are the leakage that motivated this script; the
# asset image lines are noise the model cannot see anyway.
FORBIDDEN = [
    (re.compile(r"```pin", re.I), "pin fence"),
    (re.compile(r"^\s*golden\s*:", re.I | re.M), "golden label line"),
    (re.compile(r"^\s*bbox\s*:", re.I | re.M), "bbox provenance line"),
    (re.compile(r"^!\[", re.M), "markdown image line"),
]

CORPUS_HEADER = """\
The following is the complete document library. It contains {n_docs} documents,
split into {n_chunks} numbered passages. Every passage is delimited exactly as:

[Cnnnn] Document title > Section > Subsection
<the passage text>

The bracketed identifier is that passage's address. Section paths are the
documents' own headings. Passage text is reproduced verbatim from the source
documents.
"""


def collect_chunks(index_dir: Path) -> list[dict]:
    """Every non-empty leaf, in the server's own corpus order."""
    chunks: list[dict] = []
    for path in sorted(index_dir.glob("*.json")):
        nodes = [pageindex._node_from_dict(d)
                 for d in pageindex.read_index_file(path)["nodes"]]
        by_id = pageindex._build_nodes_by_id(nodes)
        parents = pageindex._build_parent_map(nodes)
        doc_title = nodes[0].title if nodes else path.stem
        for leaf in pageindex._collect_leaves(nodes):
            content = (leaf.content or "").strip()
            if not content:
                continue
            # Ancestor chain, so a gold label naming an internal section can be
            # credited when the model returns one of that section's leaves.
            ancestors, cur = [], leaf.node_id
            while cur in parents:
                cur = parents[cur].node_id
                ancestors.append(cur)
            chunks.append({
                "chunk_id": f"C{len(chunks) + 1:04d}",
                "doc": path.stem,
                "doc_title": doc_title,
                "node_id": leaf.node_id,
                "title": leaf.title,
                "breadcrumb": pageindex._make_breadcrumb(leaf.node_id, parents, by_id),
                "ancestors": ancestors,
                "synthetic": leaf.synthetic,
                "chars": len(content),
                "content": content,
            })
    return chunks


def render(chunks: list[dict]) -> str:
    n_docs = len({c["doc"] for c in chunks})
    parts = [CORPUS_HEADER.format(n_docs=n_docs, n_chunks=len(chunks))]
    current_doc = None
    for c in chunks:
        if c["doc"] != current_doc:
            current_doc = c["doc"]
            parts.append(f"\n\n{'=' * 78}\nDOCUMENT: {c['doc_title']}\n{'=' * 78}")
        parts.append(f"\n\n[{c['chunk_id']}] {c['breadcrumb']}\n{c['content']}")
    return "".join(parts).strip() + "\n"


def audit(text: str) -> list[str]:
    problems = []
    for pattern, label in FORBIDDEN:
        hits = pattern.findall(text)
        if hits:
            problems.append(f"{len(hits)} x {label}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", default=str(ROOT / "index"))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = collect_chunks(Path(args.index_dir))
    if not chunks:
        print("no chunks found — is the index built?", file=sys.stderr)
        return 1

    text = render(chunks)
    problems = audit(text)
    if problems:
        print("REFUSING TO WRITE — forbidden content in the assembled corpus:",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 2

    (out_dir / "corpus_context.txt").write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest = {
        "sha256": sha,
        "chars": len(text),
        "n_chunks": len(chunks),
        "n_docs": len({c["doc"] for c in chunks}),
        "index_dir": str(Path(args.index_dir)),
        "chunks": [{k: v for k, v in c.items() if k != "content"} for c in chunks],
    }
    (out_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"wrote {out_dir / 'corpus_context.txt'}")
    print(f"  {len(chunks)} passages from {manifest['n_docs']} documents")
    print(f"  {len(text):,} characters, sha256 {sha[:16]}")
    print(f"  rough token estimate: {len(text) // 4:,} (at 4 chars/token) to "
          f"{int(len(text) / 3.4):,} (at 3.4)")
    print(f"wrote {out_dir / 'corpus_manifest.json'} (scoring key — never sent to the model)")
    print("audit: no pin fences, golden labels, bbox lines or image lines in the corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
