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
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ollama
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)

KB_DIR = ROOT / "knowledge_base"
INDEX_DIR = ROOT / "index"


def _normalize_backend(value: Optional[str]) -> str:
    backend = (value or "ollama").strip().lower()
    return backend if backend in {"ollama", "schlaubox"} else "ollama"


LLM_BACKEND = _normalize_backend(os.getenv("LLM_BACKEND"))
DEFAULT_MODELS = {
    "ollama": "gemma3:1b",
    "schlaubox": "llama3.3:70b",
}


def _resolve_model(backend: str) -> str:
    if backend == "schlaubox":
        return (
            os.getenv("LLM_MODEL_SCHLAUBOX")
            or os.getenv("LLM_MODEL")
            or DEFAULT_MODELS[backend]
        )
    return (
        os.getenv("LLM_MODEL_OLLAMA")
        or os.getenv("LLM_MODEL")
        or DEFAULT_MODELS[backend]
    )


DEFAULT_MODEL = _resolve_model(LLM_BACKEND)
MODEL = DEFAULT_MODEL
SYNTHESIS_MODEL = DEFAULT_MODEL

SCHLAUBOX_URL = os.getenv("SCHLAUBOX_URL", "http://intern.schlaubox.de:11434")
SCHLAUBOX_TIMEOUT = int(os.getenv("SCHLAUBOX_TIMEOUT", "500"))

# ---------------------------------------------------------------------------
# Ollama instance pool — set OLLAMA_URLS=url1,url2,... for parallelism
# Each URL is an independent Ollama process; branches are distributed
# round-robin so N instances process N branches simultaneously.
# ---------------------------------------------------------------------------
def _configured_urls() -> list[str]:
    if LLM_BACKEND == "schlaubox":
        return [SCHLAUBOX_URL]
    urls = [
        u.strip()
        for u in os.getenv("OLLAMA_URLS", "http://localhost:11434").split(",")
        if u.strip()
    ]
    return urls or ["http://localhost:11434"]


def _make_client(url: str) -> ollama.Client:
    if LLM_BACKEND == "schlaubox":
        return ollama.Client(host=url, timeout=SCHLAUBOX_TIMEOUT)
    return ollama.Client(host=url)


OLLAMA_URLS: list[str] = _configured_urls()
_clients: list[ollama.Client] = [_make_client(url) for url in OLLAMA_URLS]
_rr_lock  = threading.Lock()
_rr_index = 0

# Per-instance activity counters  {url: active_leaf_count}
_activity: dict[str, int] = {url: 0 for url in OLLAMA_URLS}
_activity_lock = threading.Lock()

# Run-level progress counter
_prog_lock  = threading.Lock()
_prog_total = 0
_prog_done  = 0

# Live event tracking for circle-pack animation
_event_lock      = threading.Lock()
_pruned_ids:     set[str] = set()   # section-pruned (and their descendants)
_retrieved_ids:  set[str] = set()   # leaves whose LLM verdict was "relevant"
_kept_ids:       set[str] = set()   # sections that passed the section check
_rejected_ids:   set[str] = set()   # leaves evaluated but verdict was "not relevant"
# Per-node decision detail, populated live as verdicts are made, so the doc
# viewer can show reasons/quotes in real time mid-run: {node_id: {status, reason, quote}}
_live_meta:      dict[str, dict] = {}


def _inc(url: str) -> None:
    with _activity_lock:
        _activity[url] = _activity.get(url, 0) + 1


def _dec(url: str) -> None:
    with _activity_lock:
        _activity[url] = max(0, _activity.get(url, 0) - 1)


def _inc_done() -> None:
    global _prog_done
    with _prog_lock:
        _prog_done += 1


def get_activity() -> dict[str, int]:
    with _activity_lock:
        return dict(_activity)


def start_run(total_leaves: int) -> None:
    global _prog_total, _prog_done
    with _prog_lock:
        _prog_total = total_leaves
        _prog_done  = 0
    with _event_lock:
        _pruned_ids.clear()
        _retrieved_ids.clear()
        _kept_ids.clear()
        _rejected_ids.clear()
        _live_meta.clear()


def get_progress() -> dict[str, int]:
    with _prog_lock:
        return {"total": _prog_total, "done": _prog_done}


def _mark_pruned(node_id: str) -> None:
    with _event_lock:
        _pruned_ids.add(node_id)


def _mark_retrieved(node_id: str) -> None:
    with _event_lock:
        _retrieved_ids.add(node_id)


def _mark_kept(node_id: str) -> None:
    with _event_lock:
        _kept_ids.add(node_id)


def _mark_rejected(node_id: str) -> None:
    with _event_lock:
        _rejected_ids.add(node_id)


def _set_live_meta(node_id: str, status: str, reason: str = "", quote: str = "") -> None:
    """Record a node's decision detail as it happens, for live doc-viewer hover."""
    with _event_lock:
        _live_meta[node_id] = {"status": status, "reason": reason, "quote": quote}


def get_live_events() -> dict:
    with _event_lock:
        return {
            "pruned":     list(_pruned_ids),
            "retrieved":  list(_retrieved_ids),
            "kept":       list(_kept_ids),
            "rejected":   list(_rejected_ids),
            "meta":       {k: dict(v) for k, v in _live_meta.items()},
        }


def _round_robin_client() -> tuple[ollama.Client, str]:
    global _rr_index
    with _rr_lock:
        idx    = _rr_index % len(_clients)
        client = _clients[idx]
        url    = OLLAMA_URLS[idx]
        _rr_index += 1
    return client, url


def reconfigure_clients(urls: list[str]) -> None:
    """Hot-swap the Ollama client pool. Called by the server when instance count changes."""
    global OLLAMA_URLS, _clients, _rr_index
    if LLM_BACKEND == "schlaubox":
        urls = [SCHLAUBOX_URL]
    elif not urls:
        urls = ["http://localhost:11434"]
    with _rr_lock:
        OLLAMA_URLS = urls
        _clients = [_make_client(u) for u in urls]
        _rr_index = 0
    with _activity_lock:
        _activity.clear()
        for u in urls:
            _activity[u] = 0


def get_backend() -> str:
    return LLM_BACKEND

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# ---------------------------------------------------------------------------
# Provenance pins — fenced ```pin blocks emitted by the betteringest_pdf
# ingest module (inline YAML: page, bbox, regions, asset info). They are part
# of the knowledge_base markdown (one render path in the viewer), but must be
# structured metadata here: stripped from all LLM-visible content and lifted
# onto the node as node.pin. Documents without pins are untouched.
# ---------------------------------------------------------------------------

PIN_FENCE_OPEN = "```pin"
PIN_BLOCK_RE = re.compile(r"^```pin\s*$\n(.*?)^```\s*$\n?", re.MULTILINE | re.DOTALL)
IMAGE_LINE_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)[ \t]*$", re.MULTILINE)


def _parse_pin_yaml(body: str) -> dict:
    """Minimal parser for the pin blocks' flat YAML: `key: scalar` lines,
    with bracketed values ([..] flow sequences) parsed as JSON."""
    pin: dict = {}
    for line in body.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("["):
            try:
                pin[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        try:
            pin[key] = int(value)
        except ValueError:
            try:
                pin[key] = float(value)
            except ValueError:
                pin[key] = value
    return pin


def _strip_pins(text: str) -> str:
    """Remove pin blocks and standalone image lines from LLM-visible content
    (the viewer still renders them from the raw markdown)."""
    text = PIN_BLOCK_RE.sub("", text)
    text = IMAGE_LINE_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

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
    pin: Optional[dict] = None          # provenance pin (page/bbox/asset) if ingested with one

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _clean_node_id(raw: str) -> str:
    """Sanitise and extract the node_id from raw strings returned by the LLM."""
    raw = raw.strip()
    if "|" in raw:
        for part in raw.split("|"):
            part = part.strip()
            if part.startswith("id="):
                return part[3:].strip()
    raw = re.sub(r"^(?:LEAF|SECTION|\[LEAF\]|\[SECTION\])[\s\\:]*", "", raw, flags=re.IGNORECASE)
    if "/" in raw:
        raw = raw.split("/")[-1]
    if raw.startswith("id="):
        raw = raw[3:]
    return raw.strip()



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
    return _strip_pins("\n".join(chunk).strip())


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
                    pin=node.pin,  # overview holds the parent's text → same location
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
    return _strip_pins("\n".join(chunk).strip())


def _populate_content(nodes: list[PageNode], lines: list[str]) -> None:
    """Recursively populate content on all real (non-synthetic) leaf nodes."""
    for node in nodes:
        if node.is_leaf and not node.synthetic:
            node.content = _extract_leaf_content(node.line_idx, node.heading_level, lines)
        elif node.children:
            _populate_content(node.children, lines)


# ---------------------------------------------------------------------------
# Pin attachment
# ---------------------------------------------------------------------------

def _attach_pins(nodes: list[PageNode], lines: list[str]) -> None:
    """Attach each heading's ```pin block (the massager places it immediately
    under the heading, before any other content or child heading) to its node.
    Runs before preamble promotion so synthetic overview leaves can inherit."""
    for node in nodes:
        i = node.line_idx + 1
        while i < len(lines) and not HEADING_RE.match(lines[i]):
            stripped = lines[i].strip()
            if stripped == PIN_FENCE_OPEN:
                body: list[str] = []
                i += 1
                while i < len(lines) and lines[i].strip() != "```":
                    body.append(lines[i])
                    i += 1
                node.pin = _parse_pin_yaml("\n".join(body))
                break
            if stripped and not stripped.startswith("```"):
                break  # real content before any pin → this heading has none
            i += 1
        _attach_pins(node.children, lines)


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
        "pin": node.pin,
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
        pin=d.get("pin"),
    )


# ---------------------------------------------------------------------------
# TOC formatting
# ---------------------------------------------------------------------------

def _flatten_toc(nodes: list[PageNode], depth: int = 0) -> str:
    """
    Sections: single line  {indent}SECTION | id={nodeId} | {title}
    Leaves:   header line + full content indented below
    Full content lets the LLM read the actual text before deciding and quote verbatim.
    """
    lines = []
    indent = "  " * depth
    for node in nodes:
        if node.is_leaf:
            lines.append(f"{indent}LEAF | id={node.node_id} | {node.title}")
            if node.content:
                lines.append(f"{indent}  {node.content}")
        else:
            lines.append(f"{indent}SECTION | id={node.node_id} | {node.title}")
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

LEAF_EVAL_PROMPT = """\
Decide if the section below directly answers the query. Read the full content carefully.

Query: QUERY_PLACEHOLDER

Document: DOCNAME_PLACEHOLDER
Section path: BREADCRUMB_PLACEHOLDER
Parent section: PARENT_SUMMARY_PLACEHOLDER

--- SECTION CONTENT ---
CONTENT_PLACEHOLDER
--- END ---

Rules:
- Answer YES only if the content directly and specifically addresses the query.
- A section that is tangentially related or only mentions the topic in passing is NOT relevant.
- If YES, the quote must be copied character-for-character from the content above.

Output exactly one of:
  {"relevant": true,  "reason": "1-2 sentences: what in this content answers the query", "quote": "1-2 verbatim sentences from the content that best answer the query"}
  {"relevant": false}

Output only the JSON object. No other text.
"""

SECTION_CHECK_PROMPT = """\
Could the document section below contain content that directly answers the query?

Query: QUERY_PLACEHOLDER
Section: BREADCRUMB_PLACEHOLDER

This section and all of its nested subsections / headings:
DESCENDANTS_PLACEHOLDER

Rules:
- Answer in JSON format.
- If ANY of the headings above might cover content that answers the query, answer:
  {"relevant": true, "reason": "1-2 sentences explaining why this section was selected"}
  The "reason" field is STRICTLY REQUIRED when "relevant" is true.
- If you are certain that none of these headings could contain a direct answer, answer:
  {"relevant": false}

Output only the JSON object. No other text.
"""

# On-demand explanation for a section that was NOT selected (rejected or pruned).
# Anchored to the actual content so the negative is verifiable, not confabulated:
# the model must describe what the section factually covers, then judge fit.
EXPLAIN_PROMPT = """\
A section of a document was not selected as answering a query. Read the section
content below and explain, factually, what it actually covers.

Query: QUERY_PLACEHOLDER
Document: DOCNAME_PLACEHOLDER
Section path: BREADCRUMB_PLACEHOLDER

--- SECTION CONTENT ---
CONTENT_PLACEHOLDER
--- END ---

First describe the section's actual topic, grounded strictly in the content above.
Then judge whether it directly answers the query.

Output exactly this JSON object:
  {"topic": "one factual sentence describing what this section actually covers", "addresses_query": false, "reason": "one sentence contrasting the section's topic with what the query asks for"}

Set "addresses_query" to true ONLY if, on re-reading, the content does directly answer the query.
Output only the JSON object. No other text.
"""


def _chat(prompt: str, client: Optional[ollama.Client] = None, url: str = "") -> str:
    if client is None:
        client, url = _round_robin_client()
    response = client.chat(
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


# ---------------------------------------------------------------------------
# Breadcrumb helpers
# ---------------------------------------------------------------------------

def _build_parent_map(nodes: list["PageNode"], parent: Optional["PageNode"] = None) -> dict:
    result: dict[str, "PageNode"] = {}
    for node in nodes:
        if parent is not None:
            result[node.node_id] = parent
        result.update(_build_parent_map(node.children, node))
    return result


def _build_nodes_by_id(nodes: list["PageNode"]) -> dict:
    result: dict[str, "PageNode"] = {}
    for node in nodes:
        result[node.node_id] = node
        result.update(_build_nodes_by_id(node.children))
    return result


def _make_breadcrumb(node_id: str, parent_map: dict, nodes_by_id: dict) -> str:
    parts: list[str] = []
    current = node_id
    while current in parent_map:
        p = parent_map[current]
        parts.append(p.title)
        current = p.node_id
    parts.reverse()
    if node_id in nodes_by_id:
        parts.append(nodes_by_id[node_id].title)
    return " > ".join(parts)


# ---------------------------------------------------------------------------
# Per-leaf evaluation
# ---------------------------------------------------------------------------

def _evaluate_leaf(
    leaf: "PageNode",
    query: str,
    doc_name: str,
    breadcrumb: str,
    parent_summary: str,
    client: Optional[ollama.Client] = None,
    client_url: str = "",
) -> tuple[str, dict]:
    """Evaluate one leaf. Returns (node_id, {relevant, reason, quote, status})."""
    prompt = (
        LEAF_EVAL_PROMPT
        .replace("QUERY_PLACEHOLDER", query)
        .replace("DOCNAME_PLACEHOLDER", doc_name.replace("_", " ").title())
        .replace("BREADCRUMB_PLACEHOLDER", breadcrumb)
        .replace("PARENT_SUMMARY_PLACEHOLDER", parent_summary or "top-level section")
        .replace("CONTENT_PLACEHOLDER", leaf.content or "(no content)")
    )
    _inc(client_url)
    try:
        raw = _chat(prompt, client, client_url)
        result = _parse_json_response(raw)
        
        if isinstance(result, dict):
            relevant = bool(result.get("relevant"))
            reason = str(result.get("reason") or "").strip()
            quote = str(result.get("quote") or "").strip()
            
            if relevant:
                _mark_retrieved(leaf.node_id)
                _set_live_meta(leaf.node_id, "retrieved", reason, quote)
                return leaf.node_id, {
                    "relevant": True,
                    "reason": reason,
                    "quote": quote,
                    "status": "retrieved"
                }
            else:
                _mark_rejected(leaf.node_id)
                _set_live_meta(leaf.node_id, "rejected", reason)
                return leaf.node_id, {
                    "relevant": False,
                    "reason": reason,
                    "quote": "",
                    "status": "rejected"
                }
    except Exception as exc:
        print(f"    [warn] leaf eval failed for {leaf.node_id}: {exc}", file=sys.stderr)
    finally:
        _dec(client_url)
    
    _mark_rejected(leaf.node_id)
    _set_live_meta(leaf.node_id, "rejected")
    return leaf.node_id, {
        "relevant": False,
        "reason": "",
        "quote": "",
        "status": "rejected"
    }


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
    _attach_pins(nodes, lines)
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


def _format_descendant_outline(node: "PageNode", depth: int = 0) -> str:
    """Indented bullet list of all descendant headings — gives the LLM full visibility."""
    lines: list[str] = []
    for child in node.children:
        prefix = "  " * depth
        marker = "•" if not child.is_leaf else "-"
        lines.append(f"{prefix}{marker} {child.title}")
        if child.children:
            lines.append(_format_descendant_outline(child, depth + 1))
    return "\n".join(l for l in lines if l)


def _check_section_relevant(
    node: "PageNode",
    query: str,
    breadcrumb: str,
    client: Optional[ollama.Client] = None,
    client_url: str = "",
) -> tuple[bool, str]:
    """Lightweight LLM call: can this section contain a direct answer?"""
    descendants = _format_descendant_outline(node) or "(no subsections)"
    prompt = (
        SECTION_CHECK_PROMPT
        .replace("QUERY_PLACEHOLDER", query)
        .replace("BREADCRUMB_PLACEHOLDER", breadcrumb)
        .replace("DESCENDANTS_PLACEHOLDER", descendants)
    )
    _inc(client_url)
    try:
        raw = _chat(prompt, client, client_url)
        result = _parse_json_response(raw)
        verdict = isinstance(result, dict) and bool(result.get("relevant"))
        reason = ""
        if isinstance(result, dict):
            reason = str(result.get("reason") or "").strip()
        
        if verdict:
            _mark_kept(node.node_id)
            _set_live_meta(node.node_id, "kept", reason)

        print(f"    [prune-check] {node.node_id}: {'KEEP' if verdict else 'PRUNE'} (raw={raw!r:.120})", file=sys.stderr)
        return verdict, reason
    except Exception as exc:
        _mark_kept(node.node_id)
        print(f"    [prune-check] {node.node_id}: ERROR ({exc}) — keeping", file=sys.stderr)
        return True, f"Error checking section: {exc}"  # conservative: never prune on error
    finally:
        _dec(client_url)


def make_client(url: str) -> ollama.Client:
    """Build an Ollama client for a specific instance URL (used by the explainer)."""
    return ollama.Client(host=url)


def explain_nonselection(
    node: "PageNode",
    query: str,
    doc_name: str,
    breadcrumb: str,
    client: Optional[ollama.Client] = None,
    client_url: str = "",
) -> dict:
    """On-demand: read the actual content and explain (grounded) what this
    non-selected section covers and why it doesn't answer the query.

    For a section with no own content, concatenate descendant leaf text so the
    judgement is anchored to real content rather than the heading alone.
    """
    content = node.content
    if not content:
        leaves = _collect_leaves([node])
        content = "\n\n".join((l.title + "\n" + (l.content or "")) for l in leaves).strip()
    content = (content or "(no content)")[:4000]

    prompt = (
        EXPLAIN_PROMPT
        .replace("QUERY_PLACEHOLDER", query)
        .replace("DOCNAME_PLACEHOLDER", doc_name.replace("_", " ").title())
        .replace("BREADCRUMB_PLACEHOLDER", breadcrumb)
        .replace("CONTENT_PLACEHOLDER", content)
    )
    try:
        raw = _chat(prompt, client, client_url)
        result = _parse_json_response(raw)
        if isinstance(result, dict):
            return {
                "topic": str(result.get("topic") or "").strip(),
                "addresses_query": bool(result.get("addresses_query")),
                "reason": str(result.get("reason") or "").strip(),
            }
    except Exception as exc:
        print(f"    [explain] failed for {node.node_id}: {exc}", file=sys.stderr)
    return {"topic": "", "addresses_query": False, "reason": ""}


def _collect_all_nodes(nodes: list["PageNode"]) -> list["PageNode"]:
    result: list["PageNode"] = []
    for node in nodes:
        result.append(node)
        result.extend(_collect_all_nodes(node.children))
    return result


def _prune_and_collect(
    nodes: list["PageNode"],
    query: str,
    parent_map: dict,
    nodes_by_id: dict,
    node_assignment: dict[str, tuple],
    node_meta: dict[str, dict],
) -> list["PageNode"]:
    """
    BFS top-down pruning. At each level, check all internal nodes in parallel;
    recurse only into relevant ones. Leaves always survive to full evaluation.
    Returns the list of leaf candidates that passed pruning.
    """
    candidate_leaves: list["PageNode"] = []
    frontier = list(nodes)

    while frontier:
        sections = [n for n in frontier if not n.is_leaf]
        candidate_leaves.extend(n for n in frontier if n.is_leaf)

        if not sections:
            break

        with ThreadPoolExecutor(max_workers=len(sections)) as pool:
            futures = {
                pool.submit(
                    _check_section_relevant,
                    node, query,
                    _make_breadcrumb(node.node_id, parent_map, nodes_by_id),
                    *node_assignment.get(node.node_id, (_clients[0], OLLAMA_URLS[0])),
                ): node
                for node in sections
            }
            next_frontier: list["PageNode"] = []
            for future in as_completed(futures):
                node = futures[future]
                verdict, reason = future.result()
                
                node_meta[node.node_id] = {
                    "relevant": verdict,
                    "reason": reason,
                    "status": "kept" if verdict else "pruned"
                }

                if verdict:
                    next_frontier.extend(node.children)
                else:
                    # Pruned: count all leaf descendants toward "done" so the
                    # progress counter reaches total. They won't be evaluated again.
                    _prune_reason = f"Pruned because ancestor section '{node.title}' was pruned."
                    _set_live_meta(node.node_id, "pruned", reason or _prune_reason)
                    for leaf in _collect_leaves([node]):
                        _inc_done()
                        _mark_pruned(leaf.node_id)
                        _set_live_meta(leaf.node_id, "pruned", _prune_reason)
                        node_meta[leaf.node_id] = {
                            "relevant": False,
                            "reason": _prune_reason,
                            "status": "pruned"
                        }
                    for descendant in _collect_all_nodes([node]):
                        _mark_pruned(descendant.node_id)
                        if descendant.node_id != node.node_id:
                            _set_live_meta(descendant.node_id, "pruned", _prune_reason)
                            node_meta[descendant.node_id] = {
                                "relevant": False,
                                "reason": _prune_reason,
                                "status": "pruned"
                            }

        frontier = next_frontier

    return candidate_leaves


def retrieve(doc_name: str, query: str) -> list[PageNode]:
    """Return relevant leaf nodes for a query against one document's index."""
    nodes_result, _ = retrieve_with_metadata(doc_name, query)
    return nodes_result


def retrieve_with_metadata(doc_name: str, query: str) -> tuple[list[PageNode], dict[str, dict]]:
    """
    Two-phase retrieval with top-down pruning.
    Phase 1: BFS section checks — prune branches whose sections are irrelevant.
    Phase 2: Full leaf evaluation (reason + quote) on survivors only.
    Returns (selected_leaves_in_doc_order, {node_id: {reason, quote}}).
    """
    index_path = INDEX_DIR / f"{doc_name}.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}. Run build first.")

    nodes = [_node_from_dict(d) for d in json.loads(index_path.read_text(encoding="utf-8"))]
    leaves = _collect_leaves(nodes)

    if not leaves:
        return [], {}

    parent_map  = _build_parent_map(nodes)
    nodes_by_id = _build_nodes_by_id(nodes)

    # Assign every node (section + leaf) to an Ollama instance round-robin by branch.
    branch_roots = nodes[0].children if (len(nodes) == 1 and nodes[0].children) else nodes
    node_assignment: dict[str, tuple[ollama.Client, str]] = {}
    for i, branch in enumerate(branch_roots):
        idx = i % len(_clients)
        client, url = _clients[idx], OLLAMA_URLS[idx]
        for node in _collect_all_nodes([branch]):
            node_assignment[node.node_id] = (client, url)
    # Fallback for any node not covered (e.g., single-root flat doc)
    for node in _collect_all_nodes(nodes):
        if node.node_id not in node_assignment:
            node_assignment[node.node_id] = (_clients[0], OLLAMA_URLS[0])

    node_meta: dict[str, dict] = {}
    if nodes:
        node_meta[nodes[0].node_id] = {
            "relevant": True,
            "reason": f"Document root '{nodes[0].title}' — starting point for retrieval.",
            "status": "kept"
        }

    # Phase 1 — top-down pruning
    top = nodes[0].children if (len(nodes) == 1 and nodes[0].children) else nodes
    surviving = _prune_and_collect(top, query, parent_map, nodes_by_id, node_assignment, node_meta)
    print(
        f"  [prune] {doc_name}: {len(surviving)}/{len(leaves)} leaves after pruning",
        file=sys.stderr,
    )

    # Phase 2 — full leaf evaluation on survivors
    if surviving:
        with ThreadPoolExecutor(max_workers=len(surviving)) as pool:
            futures = {
                pool.submit(
                    _evaluate_leaf,
                    leaf,
                    query,
                    doc_name,
                    _make_breadcrumb(leaf.node_id, parent_map, nodes_by_id),
                    parent_map[leaf.node_id].summary if leaf.node_id in parent_map else "",
                    *node_assignment.get(leaf.node_id, (_clients[0], OLLAMA_URLS[0])),
                ): leaf
                for leaf in surviving
            }
            for future in as_completed(futures):
                node_id, meta = future.result()
                node_meta[node_id] = meta
                _inc_done()

    # Preserve original document order
    selected = [l for l in leaves if node_meta.get(l.node_id, {}).get("relevant")]
    return selected, node_meta


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
