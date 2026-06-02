"""
Module 2: PageIndex
Deterministic heading-based indexing + LLM retrieval.

Build: parses markdown heading hierarchy, promotes preamble text to synthetic leaf nodes,
       extracts leaf content by heading-text matching, generates heuristic summaries.
Retrieve: sends TOC to Ollama, gets nodeIds back, returns leaf PageNode objects with content.

Usage:
    python pageindex.py              # build all docs in knowledge_base/
    python pageindex.py --doc <name> # build one doc (stem only, no .md)
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ollama

KB_DIR = Path("knowledge_base")
INDEX_DIR = Path("index")
MODEL = "gemma3:4b"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class PageNode:
    node_id: str
    title: str
    heading_level: int
    line_idx: int          # line number of heading in source (0-based)
    summary: str
    children: list["PageNode"] = field(default_factory=list)
    content: Optional[str] = None
    synthetic: bool = False
    parent_title: Optional[str] = None  # set on synthetic leaves

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _parse_headings(text: str) -> list[tuple[int, int, str, str]]:
    """
    Return [(line_idx, level, title, node_id), ...] for every heading in text.
    Duplicate heading titles are deduplicated with a -2, -3, ... suffix.
    """
    lines = text.split("\n")
    seen: dict[str, int] = {}
    headings: list[tuple[int, int, str, str]] = []

    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        base = _slugify(title)
        if base in seen:
            seen[base] += 1
            node_id = f"{base}-{seen[base]}"
        else:
            seen[base] = 1
            node_id = base
        headings.append((i, level, title, node_id))

    return headings


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------

def _build_tree(
    headings: list[tuple[int, int, str, str]],
    start: int = 0,
    end: Optional[int] = None,
) -> list[PageNode]:
    """
    Recursively build a PageNode tree from headings[start:end].
    Children of a node are all headings with strictly greater level within its span.
    """
    if end is None:
        end = len(headings)
    nodes: list[PageNode] = []
    i = start
    while i < end:
        line_idx, level, title, node_id = headings[i]
        # span: extend j forward while headings have strictly greater level
        j = i + 1
        while j < end and headings[j][1] > level:
            j += 1
        children = _build_tree(headings, i + 1, j) if j > i + 1 else []
        node = PageNode(
            node_id=node_id,
            title=title,
            heading_level=level,
            line_idx=line_idx,
            summary="",
            children=children,
        )
        nodes.append(node)
        i = j
    return nodes


# ---------------------------------------------------------------------------
# Preamble promotion
# ---------------------------------------------------------------------------

def _extract_preamble(parent_line_idx: int, first_child_line_idx: int, lines: list[str]) -> str:
    """Text between a heading line and its first child heading — stripped of blanks."""
    chunk = []
    for line in lines[parent_line_idx + 1 : first_child_line_idx]:
        if not HEADING_RE.match(line):
            chunk.append(line)
    return "\n".join(chunk).strip()


def _promote_preambles(nodes: list[PageNode], lines: list[str]) -> None:
    """
    Walk the tree in-place. For each internal node that has non-blank preamble text
    (text between the heading line and the first child heading), insert a synthetic
    leaf as the first child to hold that content.
    """
    for node in nodes:
        if node.children:
            first_child_line = node.children[0].line_idx
            preamble = _extract_preamble(node.line_idx, first_child_line, lines)
            if preamble:
                synthetic = PageNode(
                    node_id=f"{node.node_id}-overview",
                    title=f"{node.title} — Overview",
                    heading_level=node.heading_level + 1,
                    line_idx=node.line_idx,  # points to parent heading
                    summary="",
                    content=preamble,
                    synthetic=True,
                    parent_title=node.title,
                )
                node.children.insert(0, synthetic)
            # Recurse
            _promote_preambles(node.children, lines)


# ---------------------------------------------------------------------------
# Leaf content extraction
# ---------------------------------------------------------------------------

def _extract_leaf_content(line_idx: int, heading_level: int, lines: list[str]) -> str:
    """
    All lines from line_idx+1 until the next heading at level ≤ heading_level.
    Heading lines within the block are excluded (should not exist for a real leaf,
    but guarded for safety).
    """
    chunk = []
    for line in lines[line_idx + 1 :]:
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) <= heading_level:
            break
        if not m:  # exclude any stray heading lines inside the block
            chunk.append(line)
    return "\n".join(chunk).strip()


def _populate_content(nodes: list[PageNode], lines: list[str]) -> None:
    """Recursively populate content on all real (non-synthetic) leaf nodes."""
    for node in nodes:
        if node.is_leaf and not node.synthetic:
            node.content = _extract_leaf_content(node.line_idx, node.heading_level, lines)
        elif node.children:
            _populate_content(node.children, lines)


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def _generate_summary(node: PageNode) -> str:
    if node.is_leaf:
        words = (node.content or "").split()
        return " ".join(words[:25]) + ("..." if len(words) > 25 else "")
    else:
        titles = [c.title for c in node.children if not c.synthetic][:4]
        extra = node.children[4:] if len(node.children) > 4 else []
        summary = "Covers: " + ", ".join(titles)
        if extra:
            summary += f", and {len(extra)} more"
        return summary


def _populate_summaries(nodes: list[PageNode]) -> None:
    """Bottom-up summary generation (leaves first, then internal)."""
    for node in nodes:
        if node.children:
            _populate_summaries(node.children)
        node.summary = _generate_summary(node)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _node_to_dict(node: PageNode) -> dict:
    return {
        "nodeId": node.node_id,
        "title": node.title,
        "headingLevel": node.heading_level,
        "lineIdx": node.line_idx,
        "summary": node.summary,
        "isLeaf": node.is_leaf,
        "synthetic": node.synthetic,
        "parentTitle": node.parent_title,
        "content": node.content,
        "children": [_node_to_dict(c) for c in node.children],
    }


def _node_from_dict(d: dict) -> PageNode:
    children = [_node_from_dict(c) for c in d.get("children", [])]
    return PageNode(
        node_id=d["nodeId"],
        title=d["title"],
        heading_level=d["headingLevel"],
        line_idx=d["lineIdx"],
        summary=d.get("summary", ""),
        children=children,
        content=d.get("content"),
        synthetic=d.get("synthetic", False),
        parent_title=d.get("parentTitle"),
    )


# ---------------------------------------------------------------------------
# TOC formatting
# ---------------------------------------------------------------------------

def _flatten_toc(nodes: list[PageNode], depth: int = 0) -> str:
    """
    Format each node as:  {indent}LEAF|SECTION | id={nodeId} | {title} | {summary}
    The `id=` tag makes it unambiguous what to return.
    """
    lines = []
    indent = "  " * depth
    for node in nodes:
        kind = "LEAF" if node.is_leaf else "SECTION"
        lines.append(f"{indent}{kind} | id={node.node_id} | {node.title} | {node.summary}")
        if node.children:
            lines.append(_flatten_toc(node.children, depth + 1))
    return "\n".join(l for l in lines if l)


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------

def _collect_leaves(nodes: list[PageNode]) -> list[PageNode]:
    leaves: list[PageNode] = []
    for node in nodes:
        if node.is_leaf:
            leaves.append(node)
        else:
            leaves.extend(_collect_leaves(node.children))
    return leaves


def _find_nodes_by_ids(nodes: list[PageNode], ids: set[str]) -> list[PageNode]:
    """
    For each node whose node_id is in ids:
      - if leaf: return it
      - if internal: return all its leaf descendants
    Non-matching nodes are searched recursively.
    Results are deduplicated by node_id.
    """
    seen: set[str] = set()
    result: list[PageNode] = []

    def _walk(node_list: list[PageNode]) -> None:
        for node in node_list:
            if node.node_id in ids:
                if node.is_leaf:
                    if node.node_id not in seen:
                        seen.add(node.node_id)
                        result.append(node)
                else:
                    for leaf in _collect_leaves(node.children):
                        if leaf.node_id not in seen:
                            seen.add(leaf.node_id)
                            result.append(leaf)
            else:
                _walk(node.children)

    _walk(nodes)
    return result


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------

RETRIEVAL_PROMPT_TEMPLATE = """\
You are a precise document retrieval assistant.

Default answer: [] — most documents have zero relevant nodes for any given query.
Only include a node if it CLEARLY and DIRECTLY answers the query.

Document: DOCNAME_PLACEHOLDER

Each TOC line format:  TYPE | id=NODE_ID | Title | Summary

Rules:
1. First decide: does this document's subject relate to the query at all?
   If the document covers a clearly different topic, output [] and stop.
2. For each LEAF: only include it when its Title or Summary explicitly addresses the query topic.
   If in doubt, exclude it — a missed node is less harmful than a false inclusion.
3. NEVER select the top-level SECTION (first line, no indentation).
4. Return a JSON array of bare NODE_IDs (value after "id="). No prose, no fences.

Good example (include):
  Query: "What sodium restriction is recommended for hypertension?"
  LEAF: id=sodium-restriction | Sodium Restriction | Sodium intake should be limited to less than 2,300 mg/day...
  → ["sodium-restriction"]

Good example (exclude whole document):
  Query: "What sodium restriction is recommended for hypertension?"
  Document: Antibiotic Stewardship (about antimicrobial prescribing and resistance)
  → []   ← correct: antibiotics are unrelated to dietary sodium

Query: QUERY_PLACEHOLDER

Table of contents:
TOC_PLACEHOLDER
"""


def _chat(prompt: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"]


def _parse_json_response(raw: str) -> object:
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    raw = raw.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
        return data
    except json.JSONDecodeError:
        pass
    for pattern in (r"(\[.*?\])", r"(\{.*?\})"):
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model response: {raw[:300]!r}")


def _clean_node_id(raw: str) -> str:
    s = raw.strip()
    if "/" in s:
        s = s.split("/")[-1]
    if " | " in s:
        for part in s.split(" | "):
            part = part.strip()
            if part.startswith("id="):
                s = part[3:]
                break
        else:
            s = s.split(" | ")[0].strip()
    for prefix in ("LEAF ", "SECTION ", "id=", "[LEAF] ", "[SECTION] "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # Strip any remaining label-like prefix word followed by colon/backslash
    s = re.sub(r"^[A-Z]+\\?:?\s*", "", s)
    s = s.strip(" :\\")
    return s


def _select_nodes_from_llm(toc_text: str, query: str, doc_name: str = "") -> list[str]:
    prompt = (
        RETRIEVAL_PROMPT_TEMPLATE
        .replace("DOCNAME_PLACEHOLDER", doc_name.replace("_", " ").title() if doc_name else "Unknown")
        .replace("QUERY_PLACEHOLDER", query)
        .replace("TOC_PLACEHOLDER", toc_text)
    )
    raw = _chat(prompt)
    result = _parse_json_response(raw)
    if isinstance(result, list):
        return [_clean_node_id(str(x)) for x in result if x]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(doc_name: str) -> None:
    """Parse and index a single document from knowledge_base/."""
    doc_path = KB_DIR / f"{doc_name}.md"
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    print(f"  Indexing {doc_name}...")
    text = doc_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    headings = _parse_headings(text)
    nodes = _build_tree(headings)
    _promote_preambles(nodes, lines)
    _populate_content(nodes, lines)
    _populate_summaries(nodes)

    INDEX_DIR.mkdir(exist_ok=True)
    index_path = INDEX_DIR / f"{doc_name}.json"
    index_path.write_text(
        json.dumps([_node_to_dict(n) for n in nodes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    leaves = _collect_leaves(nodes)
    print(f"    → {index_path} ({len(nodes)} top-level nodes, {len(leaves)} leaves)")


def retrieve(doc_name: str, query: str) -> list[PageNode]:
    """Return relevant leaf nodes for a query against one document's index."""
    index_path = INDEX_DIR / f"{doc_name}.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}. Run build first.")

    nodes = [_node_from_dict(d) for d in json.loads(index_path.read_text(encoding="utf-8"))]
    toc = _flatten_toc(nodes)
    selected_ids = set(_select_nodes_from_llm(toc, query, doc_name))

    if not selected_ids:
        return []

    return _find_nodes_by_ids(nodes, selected_ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    global MODEL
    parser = argparse.ArgumentParser(description="Build PageIndex for knowledge_base documents.")
    parser.add_argument("--doc", help="Single doc name (stem, no .md). Default: all.")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model (default: {MODEL})")
    args = parser.parse_args()

    MODEL = args.model

    if args.doc:
        build_index(args.doc)
    else:
        docs = sorted(KB_DIR.glob("*.md"))
        if not docs:
            print(f"No documents in {KB_DIR}/. Run ingest.py first.")
            sys.exit(1)
        print(f"Building index for {len(docs)} documents...")
        for p in docs:
            build_index(p.stem)
        print("\nDone.")


if __name__ == "__main__":
    main()
