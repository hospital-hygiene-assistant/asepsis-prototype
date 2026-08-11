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
import hashlib
import json
import os
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ollama

import config as app_config
import tokens as app_tokens

KB_DIR = Path("knowledge_base")
INDEX_DIR = Path("index")

# Kept as module attributes because the server and several modules read/assign
# them (`_pi.MODEL`), but the source of truth is config.runtime().
MODEL = app_config.DEFAULT_RETRIEVAL_MODEL
SYNTHESIS_MODEL = app_config.DEFAULT_SYNTHESIS_MODEL

# ---------------------------------------------------------------------------
# Ollama instance pool — set OLLAMA_URLS=url1,url2,... for parallelism
# Each URL is an independent Ollama process; branches are distributed
# round-robin so N instances process N branches simultaneously.
# ---------------------------------------------------------------------------
OLLAMA_URLS: list[str] = [
    u.strip()
    for u in os.getenv("OLLAMA_URLS", "http://localhost:11434").split(",")
    if u.strip()
]
_clients: list[ollama.Client] = [ollama.Client(host=url) for url in OLLAMA_URLS]
_rr_lock  = threading.Lock()
_rr_index = 0

# Per-instance activity counters  {url: active_leaf_count}
_activity: dict[str, int] = {url: 0 for url in OLLAMA_URLS}
_activity_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Per-run state
#
# This used to be a set of module-level globals, which meant two concurrent
# retrievals silently overwrote each other's verdicts and progress. Everything
# now lives in a RunContext threaded through the retrieval call chain; the
# module-level helpers below delegate to the "current" run so the server's
# /api/status shape is unchanged.
# ---------------------------------------------------------------------------

# Terminal states a node can end a run in.
#   kept       an internal section survived pruning
#   pruned     the model judged the section incapable of answering
#   retrieved  a leaf was evaluated and found relevant
#   rejected   a leaf was evaluated and found NOT relevant
#   deferred   a leaf was never evaluated — the agent's context budget ran out
#   error      the call failed (parse/transport); NOT a judgement
STATUSES = ("kept", "pruned", "retrieved", "rejected", "deferred", "error")


@dataclass
class BudgetState:
    """How much of the agent's context the accepted nodes consumed."""
    tokens_used: int = 0
    tokens_max: int = 0
    evaluated: int = 0
    deferred: int = 0
    capped_by: str = ""     # "", "context" or "max_evals"


class RunContext:
    """All mutable state for a single retrieval run."""

    def __init__(self, run_id: Optional[str] = None, total_leaves: int = 0):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.lock = threading.Lock()
        self.events: dict[str, set[str]] = {s: set() for s in STATUSES}
        self.meta: dict[str, dict] = {}
        self.budget = BudgetState()
        self._total = total_leaves
        self._done = 0

    # -- verdict recording --------------------------------------------------

    def mark(self, node_id: str, status: str, reason: str = "",
             quote: str = "", **extra) -> None:
        """Record a node's verdict. A node may only hold one terminal status,
        so a re-mark moves it out of its previous set."""
        with self.lock:
            for s, bucket in self.events.items():
                if s != status:
                    bucket.discard(node_id)
            self.events.setdefault(status, set()).add(node_id)
            entry = {"status": status, "reason": reason, "quote": quote}
            entry.update(extra)
            self.meta[node_id] = entry

    def get_meta(self, node_id: str) -> dict:
        with self.lock:
            return dict(self.meta.get(node_id, {}))

    # -- progress -----------------------------------------------------------

    def set_total(self, total: int) -> None:
        with self.lock:
            self._total = total

    def inc_done(self, n: int = 1) -> None:
        with self.lock:
            self._done += n

    def progress(self) -> dict:
        with self.lock:
            return {"total": self._total, "done": self._done}

    # -- snapshots ----------------------------------------------------------

    def events_snapshot(self) -> dict:
        with self.lock:
            snap = {s: sorted(ids) for s, ids in self.events.items()}
            snap["meta"] = {k: dict(v) for k, v in self.meta.items()}
            snap["budget"] = dict(self.budget.__dict__)
            return snap

    def restore(self, snapshot: dict) -> None:
        """Load a snapshot back in — used by the debug cache to replay a run
        so the visualisation renders instead of coming up empty."""
        with self.lock:
            for s in STATUSES:
                self.events[s] = set(snapshot.get(s, []))
            self.meta = {k: dict(v) for k, v in (snapshot.get("meta") or {}).items()}
            for k, v in (snapshot.get("budget") or {}).items():
                if hasattr(self.budget, k):
                    setattr(self.budget, k, v)


_runs_lock = threading.Lock()
_runs: dict[str, RunContext] = {}
_current_run: Optional[RunContext] = None


def new_run(total_leaves: int = 0, *, make_current: bool = True) -> RunContext:
    """Create a run. Background jobs pass make_current=False so they cannot
    disturb the live visualisation of a user-facing query."""
    global _current_run
    ctx = RunContext(total_leaves=total_leaves)
    with _runs_lock:
        _runs[ctx.run_id] = ctx
        # Keep the registry from growing without bound across a long session.
        if len(_runs) > 32:
            for stale in list(_runs)[:-32]:
                if _current_run is None or stale != _current_run.run_id:
                    _runs.pop(stale, None)
        if make_current:
            _current_run = ctx
    return ctx


def current_run() -> RunContext:
    global _current_run
    with _runs_lock:
        if _current_run is None:
            _current_run = RunContext()
            _runs[_current_run.run_id] = _current_run
        return _current_run


def get_run(run_id: str) -> Optional[RunContext]:
    with _runs_lock:
        return _runs.get(run_id)


def _inc(url: str) -> None:
    with _activity_lock:
        _activity[url] = _activity.get(url, 0) + 1


def _dec(url: str) -> None:
    with _activity_lock:
        _activity[url] = max(0, _activity.get(url, 0) - 1)


def get_activity() -> dict[str, int]:
    with _activity_lock:
        return dict(_activity)


# ---------------------------------------------------------------------------
# Back-compat façade over the current run (the server polls these).
# ---------------------------------------------------------------------------

def start_run(total_leaves: int) -> RunContext:
    """Begin a new user-facing run and make it the one /api/status reports on."""
    return new_run(total_leaves=total_leaves, make_current=True)


def get_progress() -> dict[str, int]:
    return current_run().progress()


def get_live_events() -> dict:
    return current_run().events_snapshot()


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
    with _rr_lock:
        OLLAMA_URLS = urls
        _clients = [ollama.Client(host=u) for u in urls]
        _rr_index = 0
    with _activity_lock:
        _activity.clear()
        for u in urls:
            _activity[u] = 0
    _reset_work_pool()


# ---------------------------------------------------------------------------
# Shared bounded worker pool
#
# Previously each fan-out built its own ThreadPoolExecutor sized to the number
# of nodes — one thread per leaf, hundreds of threads all queueing on a couple
# of Ollama instances. One process-wide pool, sized to what the instances can
# actually absorb, replaces that; document-level and node-level fan-out share
# it, so parallelising documents cannot multiply the thread count.
# ---------------------------------------------------------------------------

_pool_lock = threading.Lock()
_work_pool: Optional[ThreadPoolExecutor] = None


def pool_size() -> int:
    per = max(1, app_config.runtime().concurrency_per_instance)
    return max(1, len(_clients) * per)


def work_pool() -> ThreadPoolExecutor:
    global _work_pool
    with _pool_lock:
        if _work_pool is None:
            _work_pool = ThreadPoolExecutor(
                max_workers=pool_size(), thread_name_prefix="asepsis")
        return _work_pool


def _reset_work_pool() -> None:
    """Rebuild the pool after the instance count or concurrency changes."""
    global _work_pool
    with _pool_lock:
        old, _work_pool = _work_pool, None
    if old is not None:
        old.shutdown(wait=False)

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

    # Summary provenance — lets a rebuild reuse an unchanged node's summary
    # instead of paying for it again (see _populate_summaries).
    content_hash: Optional[str] = None
    summary_source: Optional[str] = None          # "llm" | "heuristic"
    summary_model: Optional[str] = None
    summary_prompt_version: Optional[int] = None

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

LEAF_SUMMARY_PROMPT = """\
Summarise what the passage below CONTAINS. This summary is used later to
decide whether the passage is worth reading in full, so it must describe the
passage's actual subject matter and specifics.

Section title: TITLE_PLACEHOLDER
Section path: BREADCRUMB_PLACEHOLDER

--- PASSAGE ---
CONTENT_PLACEHOLDER
--- END ---

Rules:
- 1-2 sentences, under 50 words.
- Name the specific topics, entities, drugs, values or conditions covered.
- Describe the content only. Do not evaluate it, do not answer any question,
  do not add commentary.
- Plain text. No markdown, no bullet points, no preamble.

Write the summary now:"""

SECTION_SUMMARY_PROMPT = """\
Write a summary of a document section, given summaries of its parts.

Section title: TITLE_PLACEHOLDER
Section path: BREADCRUMB_PLACEHOLDER

--- SUMMARIES OF THIS SECTION'S PARTS ---
CHILDREN_PLACEHOLDER
--- END ---

Rules:
- 1-2 sentences, under 60 words.
- State the range of topics this section covers, specifically enough that a
  reader could tell whether an answer to a question is likely to be inside it.
- Cover the breadth of the parts; do not fixate on the first one.
- Plain text. No markdown, no bullet points, no preamble.

Write the summary now:"""


def _generate_summary(node: PageNode) -> str:
    """Heuristic summary — the fallback used when no LLM is reachable at index
    time. Leaves get their opening words; sections get their child titles.

    This is what the whole index used to run on, and it is why pruning quality
    was poor: 'Covers: Dosing, Monitoring, and 3 more' carries almost no signal
    about whether an answer lives inside. The LLM path below replaces it.
    """
    if node.is_leaf:
        words = (node.content or "").split()
        return " ".join(words[:25]) + ("..." if len(words) > 25 else "")
    else:
        # Synthetic overview leaves are the section's own prose, not a titled
        # subsection, so they are not listed by title — but they must still be
        # counted consistently, which the original slice got wrong.
        real = [c for c in node.children if not c.synthetic]
        titles = [c.title for c in real[:4]]
        extra = len(real) - len(titles)
        summary = "Covers: " + ", ".join(titles) if titles else "Covers: (no subsections)"
        if extra > 0:
            summary += f", and {extra} more"
        return summary


def _summary_input(node: PageNode) -> str:
    """The exact text a node's summary is derived from.

    Hashing this is what makes rebuilds incremental: a leaf whose content is
    untouched keeps its summary, and a section is only re-summarised when one
    of its children's summaries actually changed.
    """
    if node.is_leaf:
        return f"{node.title}\n{node.content or ''}"
    parts = []
    for child in node.children:
        label = "(section prose)" if child.synthetic else child.title
        parts.append(f"- {label}: {child.summary}")
    return f"{node.title}\n" + "\n".join(parts)


def _summary_hash(node: PageNode) -> str:
    return hashlib.sha256(_summary_input(node).encode("utf-8")).hexdigest()


def _summarise_node(node: PageNode, breadcrumb: str) -> tuple[str, str]:
    """Return (summary, source). `source` is 'llm' or 'heuristic'."""
    if node.is_leaf:
        content = node.content or ""
        if not content.strip():
            return _generate_summary(node), "heuristic"
        # Keep the summariser prompt inside its own window.
        cfg_ctx = app_config.runtime().resolved_retrieval_ctx()
        max_content = max(1000, int((cfg_ctx - 512) * 3))  # chars, generous
        truncated = content[:max_content]
        prompt = (
            LEAF_SUMMARY_PROMPT
            .replace("TITLE_PLACEHOLDER", node.title)
            .replace("BREADCRUMB_PLACEHOLDER", breadcrumb or node.title)
            .replace("CONTENT_PLACEHOLDER", truncated)
        )
    else:
        children_block = "\n".join(
            f"- {'(section prose)' if c.synthetic else c.title}: "
            f"{c.summary or '(no summary)'}"
            for c in node.children
        ) or "(no subsections)"
        prompt = (
            SECTION_SUMMARY_PROMPT
            .replace("TITLE_PLACEHOLDER", node.title)
            .replace("BREADCRUMB_PLACEHOLDER", breadcrumb or node.title)
            .replace("CHILDREN_PLACEHOLDER", children_block)
        )

    raw = _chat(prompt)
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    text = re.sub(r"^```.*?$", "", text, flags=re.MULTILINE)
    text = " ".join(text.split()).strip().strip('"')
    if not text:
        return _generate_summary(node), "heuristic"
    return text, "llm"


def _populate_summaries(
    nodes: list[PageNode],
    previous: Optional[dict[str, dict]] = None,
    *,
    use_llm: bool = False,
    breadcrumb: str = "",
    stats: Optional[dict] = None,
) -> None:
    """Bottom-up summary generation (leaves first, then internal).

    With `use_llm=False` this is the original deterministic heuristic pass.
    With `use_llm=True` each node is summarised by the model — but only if its
    summary input actually changed since `previous`, so re-indexing an
    unchanged document costs zero LLM calls, and editing one leaf re-summarises
    that leaf and its ancestor chain alone.
    """
    stats = stats if stats is not None else {}
    for node in nodes:
        path = f"{breadcrumb} > {node.title}" if breadcrumb else node.title
        if node.children:
            _populate_summaries(node.children, previous, use_llm=use_llm,
                                breadcrumb=path, stats=stats)

        if not use_llm:
            node.summary = _generate_summary(node)
            continue

        node.content_hash = _summary_hash(node)
        prior = (previous or {}).get(node.node_id) or {}
        reusable = (
            prior.get("summary")
            and prior.get("contentHash") == node.content_hash
            and prior.get("summaryModel") == MODEL
            and prior.get("summaryPromptVersion") == _summary_prompt_version()
            and prior.get("summarySource") == "llm"
        )
        if reusable:
            node.summary = prior["summary"]
            node.summary_source = "llm"
            node.summary_model = prior.get("summaryModel")
            node.summary_prompt_version = prior.get("summaryPromptVersion")
            stats["reused"] = stats.get("reused", 0) + 1
            continue

        try:
            summary, source = _summarise_node(node, path)
        except Exception as exc:
            # Flagged, never silent: the index stays valid on the heuristic,
            # and the caller reports that summaries are degraded.
            summary, source = _generate_summary(node), "heuristic"
            stats.setdefault("errors", []).append(f"{node.node_id}: {exc}")

        node.summary = summary
        node.summary_source = source
        node.summary_model = MODEL if source == "llm" else None
        node.summary_prompt_version = _summary_prompt_version() if source == "llm" else None
        stats["generated" if source == "llm" else "heuristic"] = (
            stats.get("generated" if source == "llm" else "heuristic", 0) + 1)


def _summary_prompt_version() -> int:
    """Both summary prompts move together — bumping either invalidates all."""
    v = app_config.PROMPT_VERSIONS
    return v["leaf_summary"] * 1000 + v["section_summary"]


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
        "contentHash": node.content_hash,
        "summarySource": node.summary_source,
        "summaryModel": node.summary_model,
        "summaryPromptVersion": node.summary_prompt_version,
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
        content_hash=d.get("contentHash"),
        summary_source=d.get("summarySource"),
        summary_model=d.get("summaryModel"),
        summary_prompt_version=d.get("summaryPromptVersion"),
    )


# ---------------------------------------------------------------------------
# Index files
#
# v1 was a bare JSON array of nodes, with nowhere to record which model and
# prompts produced the summaries. v2 wraps it in an object carrying that
# provenance; readers still accept a bare array and report it as v1 so the
# migration check can spot a stale index rather than crashing on it.
# ---------------------------------------------------------------------------

def read_index_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"formatVersion": 1, "nodes": data}
    data.setdefault("formatVersion", 1)
    data.setdefault("nodes", [])
    return data


def index_path_for(doc_name: str) -> Path:
    return INDEX_DIR / f"{doc_name}.json"


def load_index_nodes(doc_name: str) -> list[PageNode]:
    path = index_path_for(doc_name)
    if not path.exists():
        raise FileNotFoundError(f"Index not found: {path}. Run build first.")
    return [_node_from_dict(d) for d in read_index_file(path)["nodes"]]


def index_is_stale(path: Path) -> bool:
    """True when an index predates the current format and must be rebuilt."""
    try:
        return int(read_index_file(path).get("formatVersion", 1)) < app_config.INDEX_FORMAT_VERSION
    except (OSError, json.JSONDecodeError, ValueError):
        return True


def stale_indexes(index_dir: Optional[Path] = None) -> list[str]:
    d = index_dir or INDEX_DIR
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json") if index_is_stale(p))


# ---------------------------------------------------------------------------
# TOC formatting
# ---------------------------------------------------------------------------

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


# Appended on a retry after unparseable output. Small models sometimes wrap
# JSON in prose on the first attempt and comply when told twice.
_JSON_RETRY_SUFFIX = (
    "\n\nIMPORTANT: your previous response could not be parsed. Reply with the "
    "raw JSON object ONLY — no prose, no markdown fences, no explanation.\n"
)


def _chat(prompt: str, client: Optional[ollama.Client] = None, url: str = "",
          *, kind: str = "retrieval", model: Optional[str] = None) -> str:
    """One LLM call, with an explicit context window.

    Passing `num_ctx` is not optional: without it Ollama applies its own
    default (4096) and silently truncates anything longer, which loses
    passages with no error anywhere.
    """
    if client is None:
        client, url = _round_robin_client()
    response = client.chat(
        model=model or MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=app_config.chat_options(kind),
    )
    # Real token count of the prompt we just sent — calibrates the estimator.
    try:
        app_tokens.observe(prompt, int(response.get("prompt_eval_count") or 0))
    except (TypeError, ValueError, AttributeError):
        pass
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
    ctx: Optional[RunContext] = None,
) -> tuple[str, dict]:
    """Evaluate one leaf. Returns (node_id, {relevant, reason, quote, status}).

    A call that fails or returns unparseable output is recorded as `error`,
    never as `rejected`: a transport hiccup is not a judgement that the
    passage is irrelevant, and silently downgrading it deletes evidence.
    """
    ctx = ctx or current_run()
    prompt = (
        LEAF_EVAL_PROMPT
        .replace("QUERY_PLACEHOLDER", query)
        .replace("DOCNAME_PLACEHOLDER", doc_name.replace("_", " ").title())
        .replace("BREADCRUMB_PLACEHOLDER", breadcrumb)
        .replace("PARENT_SUMMARY_PLACEHOLDER", parent_summary or "top-level section")
        .replace("CONTENT_PLACEHOLDER", leaf.content or "(no content)")
    )
    _inc(client_url)
    last_error = ""
    try:
        for attempt in range(2):
            try:
                attempt_prompt = prompt if attempt == 0 else prompt + _JSON_RETRY_SUFFIX
                raw = _chat(attempt_prompt, client, client_url)
                result = _parse_json_response(raw)
                if not isinstance(result, dict):
                    last_error = f"expected a JSON object, got {type(result).__name__}"
                    continue

                relevant = bool(result.get("relevant"))
                reason = str(result.get("reason") or "").strip()
                quote = str(result.get("quote") or "").strip()
                status = "retrieved" if relevant else "rejected"
                ctx.mark(leaf.node_id, status, reason, quote if relevant else "")
                return leaf.node_id, {
                    "relevant": relevant,
                    "reason": reason,
                    "quote": quote if relevant else "",
                    "status": status,
                }
            except Exception as exc:
                last_error = str(exc)
                print(f"    [warn] leaf eval attempt {attempt + 1} failed for "
                      f"{leaf.node_id}: {exc}", file=sys.stderr)
    finally:
        _dec(client_url)

    ctx.mark(leaf.node_id, "error", f"Evaluation failed: {last_error}")
    return leaf.node_id, {
        "relevant": False,
        "reason": f"Evaluation failed: {last_error}",
        "quote": "",
        "status": "error",
        "error": last_error,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _previous_summaries(doc_name: str) -> dict[str, dict]:
    """{node_id: {summary, contentHash, summaryModel, summaryPromptVersion}}
    from the existing index, so an unchanged node keeps its summary."""
    path = index_path_for(doc_name)
    if not path.exists():
        return {}
    try:
        data = read_index_file(path)
    except (OSError, json.JSONDecodeError):
        return {}
    if int(data.get("formatVersion", 1)) < app_config.INDEX_FORMAT_VERSION:
        return {}   # old format has no hashes — nothing is safely reusable

    out: dict[str, dict] = {}

    def walk(items):
        for d in items:
            out[d["nodeId"]] = {
                "summary": d.get("summary"),
                "contentHash": d.get("contentHash"),
                "summaryModel": d.get("summaryModel"),
                "summaryPromptVersion": d.get("summaryPromptVersion"),
                "summarySource": d.get("summarySource"),
            }
            walk(d.get("children", []))

    walk(data["nodes"])
    return out


def build_index(doc_name: str, *, use_llm_summaries: bool = True,
                progress=None) -> dict:
    """Parse and index a single document from knowledge_base/.

    Structure parsing is deterministic; summaries are LLM-generated, computed
    once here and persisted. Nodes whose summary input is unchanged since the
    last build reuse their stored summary, so re-indexing an untouched
    document makes no LLM calls at all.

    Returns a small report: counts of generated/reused summaries and any
    summariser errors (which fall back to the heuristic rather than failing).
    """
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

    stats: dict = {}
    previous = _previous_summaries(doc_name) if use_llm_summaries else {}
    if progress:
        progress({"phase": "summarise", "doc": doc_name})
    _populate_summaries(nodes, previous, use_llm=use_llm_summaries, stats=stats)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = index_path_for(doc_name)
    index_path.write_text(
        json.dumps({
            "formatVersion": app_config.INDEX_FORMAT_VERSION,
            "doc": doc_name,
            "summaryModel": MODEL if use_llm_summaries else None,
            "summaryPromptVersion": _summary_prompt_version(),
            "nodes": [_node_to_dict(n) for n in nodes],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    leaves = _collect_leaves(nodes)
    report = {
        "doc": doc_name,
        "nodes": len(_collect_all_nodes(nodes)),
        "leaves": len(leaves),
        "summaries_generated": stats.get("generated", 0),
        "summaries_reused": stats.get("reused", 0),
        "summaries_heuristic": stats.get("heuristic", 0),
        "errors": stats.get("errors", []),
    }
    print(f"    → {index_path} ({len(nodes)} top-level nodes, {len(leaves)} leaves; "
          f"{report['summaries_generated']} summaries generated, "
          f"{report['summaries_reused']} reused)")
    if report["summaries_heuristic"]:
        print(f"    [warn] {report['summaries_heuristic']} summaries fell back to the "
              f"heuristic — retrieval quality will be degraded for those nodes",
              file=sys.stderr)
    return report


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
    ctx: Optional[RunContext] = None,
) -> tuple[bool, str]:
    """Lightweight LLM call: can this section contain a direct answer?

    Errors keep the section (conservative — never prune on a failure), but are
    recorded as `error` rather than a clean `kept` so the UI can distinguish a
    decision from a failure.
    """
    ctx = ctx or current_run()
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
            ctx.mark(node.node_id, "kept", reason)

        print(f"    [prune-check] {node.node_id}: {'KEEP' if verdict else 'PRUNE'} (raw={raw!r:.120})", file=sys.stderr)
        return verdict, reason
    except Exception as exc:
        ctx.mark(node.node_id, "error",
                 f"Section check failed, section kept as a precaution: {exc}")
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
    ctx: Optional[RunContext] = None,
) -> list["PageNode"]:
    """
    BFS top-down pruning. At each level, check all internal nodes in parallel;
    recurse only into relevant ones. Leaves always survive to full evaluation.
    Returns the list of leaf candidates that passed pruning.
    """
    ctx = ctx or current_run()
    candidate_leaves: list["PageNode"] = []
    frontier = list(nodes)
    pool = work_pool()

    while frontier:
        sections = [n for n in frontier if not n.is_leaf]
        candidate_leaves.extend(n for n in frontier if n.is_leaf)

        if not sections:
            break

        futures = {
            pool.submit(
                _check_section_relevant,
                node, query,
                _make_breadcrumb(node.node_id, parent_map, nodes_by_id),
                *node_assignment.get(node.node_id, (_clients[0], OLLAMA_URLS[0])),
                ctx=ctx,
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
                "status": ctx.get_meta(node.node_id).get("status") or
                          ("kept" if verdict else "pruned"),
            }

            if verdict:
                next_frontier.extend(node.children)
            else:
                # Pruned: count all leaf descendants toward "done" so the
                # progress counter reaches total. They won't be evaluated again.
                _prune_reason = f"Pruned because ancestor section '{node.title}' was pruned."
                ctx.mark(node.node_id, "pruned", reason or _prune_reason)
                node_meta[node.node_id]["status"] = "pruned"
                for leaf in _collect_leaves([node]):
                    ctx.inc_done()
                    ctx.mark(leaf.node_id, "pruned", _prune_reason)
                    node_meta[leaf.node_id] = {
                        "relevant": False,
                        "reason": _prune_reason,
                        "status": "pruned",
                    }
                for descendant in _collect_all_nodes([node]):
                    if descendant.node_id != node.node_id:
                        ctx.mark(descendant.node_id, "pruned", _prune_reason)
                        node_meta[descendant.node_id] = {
                            "relevant": False,
                            "reason": _prune_reason,
                            "status": "pruned",
                        }

        frontier = next_frontier

    return candidate_leaves


def retrieve(doc_name: str, query: str, ctx: Optional[RunContext] = None) -> list[PageNode]:
    """Return relevant leaf nodes for a query against one document's index."""
    nodes_result, _ = retrieve_with_metadata(doc_name, query, ctx=ctx)
    return nodes_result


def retrieve_with_metadata(
    doc_name: str, query: str, ctx: Optional[RunContext] = None,
) -> tuple[list[PageNode], dict[str, dict]]:
    """
    Two-phase retrieval with top-down pruning.
    Phase 1: BFS section checks — prune branches whose sections are irrelevant.
    Phase 2: Full leaf evaluation (reason + quote) on survivors only.
    Returns (selected_leaves_in_doc_order, {node_id: {reason, quote}}).
    """
    ctx = ctx or current_run()
    nodes = load_index_nodes(doc_name)
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
    surviving = _prune_and_collect(top, query, parent_map, nodes_by_id,
                                   node_assignment, node_meta, ctx=ctx)
    print(
        f"  [prune] {doc_name}: {len(surviving)}/{len(leaves)} leaves after pruning",
        file=sys.stderr,
    )

    # Phase 2 — full leaf evaluation on survivors
    if surviving:
        pool = work_pool()
        futures = {
            pool.submit(
                _evaluate_leaf,
                leaf,
                query,
                doc_name,
                _make_breadcrumb(leaf.node_id, parent_map, nodes_by_id),
                parent_map[leaf.node_id].summary if leaf.node_id in parent_map else "",
                *node_assignment.get(leaf.node_id, (_clients[0], OLLAMA_URLS[0])),
                ctx=ctx,
            ): leaf
            for leaf in surviving
        }
        for future in as_completed(futures):
            node_id, meta = future.result()
            node_meta[node_id] = meta
            ctx.inc_done()

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
