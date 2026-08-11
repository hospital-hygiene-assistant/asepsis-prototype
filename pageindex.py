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
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ollama

import choices
import config as app_config
import ranking
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


# ---------------------------------------------------------------------------
# Coordination pool — SEPARATE from the work pool, and it must stay that way.
#
# Orchestration tasks (one per document) spend their time blocked on the
# node-level LLM calls they submit. Running them on the work pool deadlocks:
# with N workers, N documents occupy every worker, and each then waits for a
# worker that can never free up. That is not theoretical — it is exactly what
# made every query hang after documents started retrieving in parallel.
#
# INVARIANT: nothing running on WORK_POOL may submit to WORK_POOL and block.
# Coordinators run here; only leaf LLM calls run there.
# ---------------------------------------------------------------------------

_coord_lock = threading.Lock()
_coord_pool: Optional[ThreadPoolExecutor] = None


def coordination_pool() -> ThreadPoolExecutor:
    global _coord_pool
    with _coord_lock:
        if _coord_pool is None:
            # These threads block rather than compute, so the size only needs
            # to cover the corpus, not the hardware.
            _coord_pool = ThreadPoolExecutor(
                max_workers=32, thread_name_prefix="asepsis-coord")
        return _coord_pool

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
    # The separator is required. With `*` this also ate the prefix of any
    # legitimate id that merely STARTS with these words — "leaf-1" became
    # "-1", and a real heading slugged to "section-overview" became
    # "-overview", which then failed to match any node.
    raw = re.sub(r"^(?:LEAF|SECTION|\[LEAF\]|\[SECTION\])[\s\\:]+", "", raw, flags=re.IGNORECASE)
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

    raw = _chat(prompt, kind="summary", model=_summary_model())
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    text = re.sub(r"^```.*?$", "", text, flags=re.MULTILINE)
    text = " ".join(text.split()).strip().strip('"')
    if not text:
        return _generate_summary(node), "heuristic"
    return text, "llm"


def _nodes_by_depth(
    nodes: list[PageNode], depth: int = 0,
    out: Optional[dict] = None, breadcrumb: str = "",
) -> dict[int, list[tuple[PageNode, str]]]:
    """Group every node by its depth, carrying its breadcrumb."""
    out = out if out is not None else {}
    for node in nodes:
        path = f"{breadcrumb} > {node.title}" if breadcrumb else node.title
        out.setdefault(depth, []).append((node, path))
        _nodes_by_depth(node.children, depth + 1, out, path)
    return out


def _summarise_one(node: PageNode, path: str, previous: Optional[dict]) -> tuple[str, str, Optional[str]]:
    """Summarise a single node. Returns (summary, source, error)."""
    node.content_hash = _summary_hash(node)
    prior = (previous or {}).get(node.node_id) or {}
    # Compare against the model that actually WRITES summaries, which may be a
    # lighter one than the retrieval model. Comparing against MODEL would let a
    # change of summary model silently reuse the old model's summaries.
    summary_model = _summary_model()
    reusable = (
        prior.get("summary")
        and prior.get("contentHash") == node.content_hash
        and prior.get("summaryModel") == summary_model
        and prior.get("summaryPromptVersion") == _summary_prompt_version()
        and prior.get("summarySource") == "llm"
    )
    if reusable:
        return prior["summary"], "reused", None
    try:
        summary, source = _summarise_node(node, path)
        return summary, source, None
    except Exception as exc:
        # Flagged, never silent: the index stays valid on the heuristic,
        # and the caller reports that summaries are degraded.
        return _generate_summary(node), "heuristic", str(exc)


def _populate_summaries(
    nodes: list[PageNode],
    previous: Optional[dict[str, dict]] = None,
    *,
    use_llm: bool = False,
    breadcrumb: str = "",
    stats: Optional[dict] = None,
    progress=None,
) -> None:
    """Bottom-up summary generation (leaves first, then internal).

    With `use_llm=False` this is the original deterministic heuristic pass.
    With `use_llm=True` each node is summarised by the model — but only if its
    summary input actually changed since `previous`, so re-indexing an
    unchanged document costs zero LLM calls, and editing one leaf re-summarises
    that leaf and its ancestor chain alone.

    Nodes at the same depth are summarised CONCURRENTLY. A node's summary
    depends only on its children's, which are strictly deeper, so processing
    depth-by-depth from the bottom keeps the dependency order exact while
    letting each level's calls overlap. Doing this one node at a time made
    indexing take the sum of every call's latency.
    """
    stats = stats if stats is not None else {}

    if not use_llm:
        for node in nodes:
            if node.children:
                _populate_summaries(node.children, previous, use_llm=False,
                                    stats=stats)
            node.summary = _generate_summary(node)
        return

    by_depth = _nodes_by_depth(nodes, breadcrumb=breadcrumb)
    pool = work_pool()
    total = sum(len(v) for v in by_depth.values())
    done = 0

    for depth in sorted(by_depth, reverse=True):     # deepest first
        level = by_depth[depth]
        futures = [(node, pool.submit(_summarise_one, node, path, previous))
                   for node, path in level]
        for node, future in futures:
            summary, source, error = future.result()
            node.summary = summary
            if error:
                stats.setdefault("errors", []).append(f"{node.node_id}: {error}")
            summary_model = _summary_model()
            if source == "reused":
                node.summary_source = "llm"
                node.summary_model = summary_model
                node.summary_prompt_version = _summary_prompt_version()
                stats["reused"] = stats.get("reused", 0) + 1
            else:
                node.summary_source = source
                node.summary_model = summary_model if source == "llm" else None
                node.summary_prompt_version = (
                    _summary_prompt_version() if source == "llm" else None)
                key = "generated" if source == "llm" else "heuristic"
                stats[key] = stats.get(key, 0) + 1
            done += 1
            if progress:
                progress({"node": node.node_id, "title": node.title,
                          "depth": depth, "done": done, "total": total,
                          "source": source})


def _summary_model() -> str:
    """The model that writes summaries.

    `MODEL` stays the single authority for the retrieval model (the server
    assigns it, and tests patch it); config's `summary_model` is an optional
    override for pointing summarisation at something lighter, since it is a
    much easier task than the retrieval judgements.
    """
    return app_config.runtime().summary_model or MODEL


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

# The pruning prompt. One call per PARENT, showing all of its direct children
# together so the model chooses among them comparatively — the previous design
# asked about each section in isolation, which cost one call per node and gave
# the model no basis for preferring one branch over its siblings.
#
# Children are described by their index-time summaries, so this prompt never
# needs to recurse into the subtree.
CHILD_SELECT_PROMPT = """\
You are navigating a document to find content that answers a query. Below are
the direct subsections of one part of the document, each with a summary of what
it contains. Choose which ones are worth opening.

Query: QUERY_PLACEHOLDER

Currently in: BREADCRUMB_PLACEHOLDER

Subsections:
CHILDREN_PLACEHOLDER

Rules:
- Select every subsection that could plausibly contain an answer. It is much
  worse to miss a relevant subsection than to open an irrelevant one.
- Compare them against each other: prefer the ones whose summaries actually
  bear on the query.
- If none of them bear on the query at all, select none.
- Use the ids exactly as written above. Never invent an id.

Output exactly this JSON object and nothing else:
  {"keep": [{"id": "<id>", "reason": "<one short sentence>"}]}
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
    response = _chat_call(client, model or MODEL, prompt, kind)
    # Real token count of the prompt we just sent — calibrates the estimator.
    try:
        app_tokens.observe(prompt, int(response.get("prompt_eval_count") or 0))
    except (TypeError, ValueError, AttributeError):
        pass
    return response["message"]["content"]


def _chat_call(client: ollama.Client, model: str, prompt: str, kind: str):
    """One chat request, with thinking disabled where it buys nothing.

    Models that do not support the `think` parameter reject it, so the call
    falls back to a plain request rather than failing the whole run.
    """
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": app_config.chat_options(kind),
        "keep_alive": app_config.keep_alive(),
    }
    try:
        return client.chat(think=app_config.think_for(kind), **kwargs)
    except Exception as exc:
        if "think" not in str(exc).lower():
            raise
        return client.chat(**kwargs)


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
                progress=None, verbose: bool = False) -> dict:
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

    node_total = len(_collect_all_nodes(nodes))
    started = time.time()

    def _report(info: dict) -> None:
        if progress:
            progress({"phase": "summarise", "doc": doc_name, **info})
        if not verbose:
            return
        done, total = info["done"], info["total"]
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        mark = {"llm": "+", "reused": "=", "heuristic": "!"}.get(info["source"], "?")
        title = info["title"][:44]
        print(f"      [{done:>3}/{total}] {mark} {title:<44s} "
              f"{elapsed:5.1f}s elapsed, ~{eta:4.0f}s left", flush=True)

    if verbose:
        print(f"    {node_total} nodes to summarise "
              f"({len(_collect_leaves(nodes))} leaves), "
              f"{pool_size()} call(s) in flight at a time", flush=True)

    _populate_summaries(nodes, previous, use_llm=use_llm_summaries,
                        stats=stats, progress=_report)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = index_path_for(doc_name)
    index_path.write_text(
        json.dumps({
            "formatVersion": app_config.INDEX_FORMAT_VERSION,
            "doc": doc_name,
            "summaryModel": _summary_model() if use_llm_summaries else None,
            "summaryPromptVersion": _summary_prompt_version(),
            "nodes": [_node_to_dict(n) for n in nodes],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    leaves = _collect_leaves(nodes)
    report = {
        "doc": doc_name,
        "seconds": round(time.time() - started, 1),
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


# ---------------------------------------------------------------------------
# Batched child selection (the pruning decision)
# ---------------------------------------------------------------------------

def _format_children_block(children: list["PageNode"]) -> str:
    lines = []
    for child in children:
        kind = "LEAF" if child.is_leaf else "SECTION"
        title = f"{child.parent_title} (section prose)" if child.synthetic else child.title
        summary = (child.summary or "(no summary available)").strip()
        lines.append(f"- id={child.node_id} | {kind} | {title}\n    {summary}")
    return "\n".join(lines)


def _child_batches(children: list["PageNode"], breadcrumb: str, query: str) -> list[list["PageNode"]]:
    """Split a child list into groups that each fit the retrieval window.

    Splitting costs the model its full comparative view, so it only happens
    when the alternative is an overflowing prompt.
    """
    budget = app_config.runtime().resolved_retrieval_ctx()
    overhead = app_tokens.estimate_tokens(CHILD_SELECT_PROMPT + breadcrumb + query) + 512
    available = max(256, budget - overhead)

    batches, current, used = [], [], 0
    for child in children:
        cost = app_tokens.estimate_tokens(
            f"{child.node_id}{child.title}{child.summary or ''}") + 8
        if current and used + cost > available:
            batches.append(current)
            current, used = [], 0
        current.append(child)
        used += cost
    if current:
        batches.append(current)
    return batches


def _select_children(
    children: list["PageNode"],
    query: str,
    breadcrumb: str,
    client: Optional[ollama.Client] = None,
    client_url: str = "",
    ctx: Optional[RunContext] = None,
) -> tuple[dict[str, str], bool]:
    """Ask the model which of `children` are worth descending into.

    Returns ({node_id: reason}, errored). On any failure every child is kept —
    pruning on a failed call would silently delete content.
    """
    ctx = ctx or current_run()
    if not children:
        return {}, False

    batches = _child_batches(children, breadcrumb, query)
    kept: dict[str, str] = {}
    errored = False
    valid_ids = {c.node_id for c in children}

    for batch in batches:
        prompt = (
            CHILD_SELECT_PROMPT
            .replace("QUERY_PLACEHOLDER", query)
            .replace("BREADCRUMB_PLACEHOLDER", breadcrumb or "the document root")
            .replace("CHILDREN_PLACEHOLDER", _format_children_block(batch))
        )
        _inc(client_url)
        try:
            raw = _chat(prompt, client, client_url, kind="prune")
            result = _parse_json_response(raw)
            entries = result.get("keep", []) if isinstance(result, dict) else result
            if not isinstance(entries, list):
                raise ValueError(f"expected a list of kept ids, got {type(entries).__name__}")

            batch_ids = {c.node_id for c in batch}
            for entry in entries:
                if isinstance(entry, dict):
                    node_id = _clean_node_id(str(entry.get("id", "")))
                    reason = str(entry.get("reason") or "").strip()
                else:
                    node_id, reason = _clean_node_id(str(entry)), ""
                if node_id in batch_ids:
                    kept[node_id] = reason
                elif node_id in valid_ids:
                    kept[node_id] = reason  # right doc, wrong batch — accept it
                else:
                    print(f"    [prune] ignoring unknown id {node_id!r} from the model",
                          file=sys.stderr)
        except Exception as exc:
            # Conservative: this batch survives intact rather than vanishing.
            errored = True
            for child in batch:
                kept.setdefault(child.node_id, f"Kept after a failed selection call: {exc}")
            print(f"    [prune] selection failed under {breadcrumb!r}: {exc} — keeping all",
                  file=sys.stderr)
        finally:
            _dec(client_url)

    # NOTE: an empty selection is a legitimate verdict at every level,
    # including a document's top level — "this document has nothing to do with
    # the question" is exactly the judgement cross-document retrieval needs.
    # Overriding it here forced every document to be explored in full for every
    # query. The only genuinely suspicious case is EVERY document pruning to
    # nothing, which is checked once at the corpus level by the caller.
    return kept, errored


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
    pool = work_pool()

    def _prune_subtree(node: "PageNode", reason: str) -> None:
        """Record a rejected child and everything beneath it."""
        ctx.mark(node.node_id, "pruned", reason)
        node_meta[node.node_id] = {"relevant": False, "reason": reason, "status": "pruned"}
        descendant_reason = f"Pruned because ancestor section '{node.title}' was pruned."
        for descendant in _collect_all_nodes([node]):
            if descendant.node_id == node.node_id:
                continue
            ctx.mark(descendant.node_id, "pruned", descendant_reason)
            node_meta[descendant.node_id] = {
                "relevant": False, "reason": descendant_reason, "status": "pruned"}
        # Pruned leaves count toward "done" so the progress bar still reaches
        # its total — they will never be evaluated.
        for leaf in _collect_leaves([node]):
            ctx.inc_done()

    # Each unit of work is a PARENT and its direct children — one LLM call per
    # group, rather than one per node.
    groups: list[tuple[Optional["PageNode"], list["PageNode"]]] = [(None, list(nodes))]
    level = 0

    while groups:
        futures = {}
        for parent, children in groups:
            if not children:
                continue
            breadcrumb = ("" if parent is None
                          else _make_breadcrumb(parent.node_id, parent_map, nodes_by_id))
            assignment = node_assignment.get(
                parent.node_id if parent is not None else children[0].node_id,
                (_clients[0], OLLAMA_URLS[0]))
            futures[pool.submit(
                _select_children, children, query, breadcrumb,
                *assignment, ctx=ctx,
            )] = (parent, children)

        next_groups: list[tuple[Optional["PageNode"], list["PageNode"]]] = []
        for future in as_completed(futures):
            parent, children = futures[future]
            kept, errored = future.result()

            for child in children:
                if child.node_id in kept:
                    reason = kept[child.node_id]
                    if child.is_leaf:
                        # A kept leaf is only a candidate — its verdict comes
                        # from the per-leaf evaluator in the next phase.
                        candidate_leaves.append(child)
                    else:
                        status = "error" if errored else "kept"
                        ctx.mark(child.node_id, status, reason)
                        node_meta[child.node_id] = {
                            "relevant": True, "reason": reason, "status": status}
                        next_groups.append((child, child.children))
                else:
                    _prune_subtree(child, reason=(
                        kept.get(child.node_id)
                        or ("Not relevant to this question: judged against its "
                            "siblings, this branch was not worth opening."
                            if level == 0 else
                            "Not selected: the model judged this branch unlikely "
                            "to answer the query when compared with its siblings.")))

        groups = next_groups
        level += 1

    return candidate_leaves


@dataclass
class DocCandidates:
    """One document's state between pruning and leaf evaluation."""
    doc_name: str
    nodes: list["PageNode"]
    leaves: list["PageNode"]
    parent_map: dict
    nodes_by_id: dict
    node_assignment: dict
    candidates: list["PageNode"]
    node_meta: dict
    # {leaf_id: [answer_ids]} — which of the user's selections put this leaf
    # in play. Under union semantics leaves genuinely differ here, so it is
    # the informative part of the pre-filter.
    choice_tags: dict = field(default_factory=dict)


def prune_document(doc_name: str, query: str, ctx: Optional[RunContext] = None,
                   selected_answers: Optional[list[str]] = None) -> DocCandidates:
    """Phase 1 for one document: load the index and prune it top-down.

    When the user has answered the multiple-choice questions, their selections
    narrow the tree FIRST, using judgements precomputed at index time. That
    costs no LLM calls at query time — it is a set lookup — and everything
    outside the selection is recorded as pruned by the choice filter rather
    than being silently absent.
    """
    ctx = ctx or current_run()
    nodes = load_index_nodes(doc_name)
    leaves = _collect_leaves(nodes)

    choice_tags: dict[str, list[str]] = {}
    if selected_answers:
        store = choices.FacetStore(INDEX_DIR, doc_name)
        keep_leaf_ids = choices.selected_leaf_ids(store, leaves, selected_answers)
        if keep_leaf_ids:
            keep_ids = choices.ancestor_closure(nodes, keep_leaf_ids)
            for node in _collect_all_nodes(nodes):
                if node.node_id not in keep_ids:
                    reason = ("Excluded by your answers to the pre-filter questions "
                              "— this passage was not judged relevant to any of them.")
                    ctx.mark(node.node_id, "pruned", reason)
            for leaf in leaves:
                if leaf.node_id not in keep_leaf_ids:
                    ctx.inc_done()
            nodes = choices.prune_tree_to(nodes, keep_ids)
            leaves = _collect_leaves(nodes)
            for leaf in leaves:
                choice_tags[leaf.node_id] = choices.answers_for_leaf(
                    store, leaf, selected_answers)
        else:
            # An empty selection would mean querying nothing at all. Fall back
            # to the whole document and say so, rather than silently returning
            # no evidence.
            print(f"  [choices] {doc_name}: no leaves matched the selected answers "
                  f"— using the whole document", file=sys.stderr)

    parent_map = _build_parent_map(nodes)
    nodes_by_id = _build_nodes_by_id(nodes)

    # Assign every node to an Ollama instance, round-robin by branch.
    branch_roots = nodes[0].children if (len(nodes) == 1 and nodes[0].children) else nodes
    node_assignment: dict[str, tuple] = {}
    for i, branch in enumerate(branch_roots):
        idx = i % len(_clients)
        client, url = _clients[idx], OLLAMA_URLS[idx]
        for node in _collect_all_nodes([branch]):
            node_assignment[node.node_id] = (client, url)
    for node in _collect_all_nodes(nodes):
        node_assignment.setdefault(node.node_id, (_clients[0], OLLAMA_URLS[0]))

    node_meta: dict[str, dict] = {}
    if nodes:
        node_meta[nodes[0].node_id] = {
            "relevant": True,
            "reason": f"Document root '{nodes[0].title}' — starting point for retrieval.",
            "status": "kept",
        }

    candidates: list[PageNode] = []
    if leaves:
        top = nodes[0].children if (len(nodes) == 1 and nodes[0].children) else nodes
        candidates = _prune_and_collect(top, query, parent_map, nodes_by_id,
                                        node_assignment, node_meta, ctx=ctx)
        print(f"  [prune] {doc_name}: {len(candidates)}/{len(leaves)} leaves after pruning",
              file=sys.stderr)

    return DocCandidates(doc_name, nodes, leaves, parent_map, nodes_by_id,
                         node_assignment, candidates, node_meta, choice_tags)


# Rough allowance for the synthesis prompt's own scaffolding (instructions,
# format block, per-passage headers) on top of the passages themselves.
SYNTHESIS_OVERHEAD_TOKENS = 900


def agent_budget_tokens(query: str) -> int:
    """How many tokens of retrieved passages the agent can actually reason over."""
    cfg = app_config.runtime()
    spec = app_config.spec_for(cfg.synthesis_model)
    return max(0, cfg.resolved_agent_ctx() - spec.reserve
               - SYNTHESIS_OVERHEAD_TOKENS - app_tokens.estimate_tokens(query))


def _leaf_cost(leaf: "PageNode") -> int:
    """A leaf's cost against the agent's window: its passage plus its header."""
    return app_tokens.estimate_tokens(f"{leaf.title}\n{leaf.content or ''}") + 16


def evaluate_ranked(
    docs: list[DocCandidates],
    query: str,
    ctx: Optional[RunContext] = None,
) -> None:
    """Phase 2: evaluate candidate leaves in BM25 order under the agent's budget.

    Every retrieved node ends up in the agent's context, so that window — not
    the evaluator's — is the real constraint. Candidates are ranked against the
    query, then evaluated in order until the ACCEPTED passages would overflow
    it. Rejected leaves cost no budget, so the walk continues past them.

    Whatever is left unevaluated is marked `deferred`: not a judgement, just
    the tail of the ranking that the agent had no room for. It stays browsable
    in the UI.
    """
    ctx = ctx or current_run()
    candidates = [(d, leaf) for d in docs for leaf in d.candidates]
    budget = agent_budget_tokens(query)
    ctx.budget.tokens_max = budget

    if not candidates:
        return

    ranked = ranking.rank(candidates, query, text_of=lambda pair: (
        f"{pair[1].title}\n{pair[1].summary or ''}\n{pair[1].content or ''}"))

    max_evals = max(1, app_config.runtime().max_leaf_evals)
    pool = work_pool()
    wave_size = max(1, pool_size())

    accepted_tokens = 0
    evaluated = 0
    stop_reason = ""
    position = 0

    while position < len(ranked):
        if stop_reason:
            break
        wave = ranked[position:position + wave_size]
        futures = []
        for scored in wave:
            doc, leaf = scored.item
            futures.append((scored, pool.submit(
                _evaluate_leaf,
                leaf, query, doc.doc_name,
                _make_breadcrumb(leaf.node_id, doc.parent_map, doc.nodes_by_id),
                doc.parent_map[leaf.node_id].summary if leaf.node_id in doc.parent_map else "",
                *doc.node_assignment.get(leaf.node_id, (_clients[0], OLLAMA_URLS[0])),
                ctx=ctx,
            )))

        # Commit strictly in rank order, so the cutoff is deterministic no
        # matter which call happens to finish first.
        committed = 0
        for scored, future in futures:
            doc, leaf = scored.item
            node_id, meta = future.result()
            committed += 1
            meta["bm25_score"] = round(scored.score, 4)
            meta["bm25_rank"] = scored.rank
            doc.node_meta[node_id] = meta
            ctx.inc_done()
            evaluated += 1
            position += 1

            if meta.get("relevant"):
                cost = _leaf_cost(leaf)
                if accepted_tokens + cost > budget:
                    # This passage cannot fit; it and everything below it in
                    # the ranking are deferred rather than accepted.
                    meta.update({
                        "relevant": False,
                        "status": "deferred",
                        "reason": ("Judged relevant, but the agent's context budget was "
                                   "already full — this passage was not included."),
                    })
                    ctx.mark(leaf.node_id, "deferred", meta["reason"],
                             bm25_rank=scored.rank, bm25_score=meta["bm25_score"])
                    stop_reason = "context"
                    break
                accepted_tokens += cost

            if evaluated >= max_evals:
                stop_reason = "max_evals"
                break

        # A wave runs in parallel, so when we stop mid-wave the remaining
        # calls are still in flight — and each marks its own verdict on the
        # context when it lands. Drain them before recording the deferred
        # tail, or a late "retrieved" write would silently overwrite it.
        for _, future in futures[committed:]:
            try:
                future.result()
            except Exception:
                pass

    # Everything the queue never reached.
    deferred = 0
    for scored in ranked[position:]:
        doc, leaf = scored.item
        reason = ("Not evaluated: the agent's context budget was reached before this "
                  "passage's turn in the relevance ranking."
                  if stop_reason != "max_evals" else
                  "Not evaluated: the per-run evaluation cap was reached.")
        doc.node_meta[leaf.node_id] = {
            "relevant": False, "reason": reason, "status": "deferred",
            "bm25_score": round(scored.score, 4), "bm25_rank": scored.rank,
        }
        ctx.mark(leaf.node_id, "deferred", reason,
                 bm25_rank=scored.rank, bm25_score=round(scored.score, 4))
        ctx.inc_done()
        deferred += 1

    ctx.budget.tokens_used = accepted_tokens
    ctx.budget.evaluated = evaluated
    ctx.budget.deferred = deferred + (1 if stop_reason == "context" else 0)
    ctx.budget.capped_by = stop_reason

    if stop_reason:
        print(f"  [budget] stopped after {evaluated} evaluations "
              f"({accepted_tokens}/{budget} tokens accepted); "
              f"{ctx.budget.deferred} passage(s) deferred [{stop_reason}]",
              file=sys.stderr)


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

    Single-document convenience wrapper — the budget applies within this
    document alone. The server uses prune_document + evaluate_ranked directly
    so that ranking and the agent's budget span the whole corpus.
    """
    ctx = ctx or current_run()
    doc = prune_document(doc_name, query, ctx=ctx)
    if not doc.leaves:
        return [], {}

    evaluate_ranked([doc], query, ctx=ctx)

    # Preserve original document order
    selected = [l for l in doc.leaves if doc.node_meta.get(l.node_id, {}).get("relevant")]
    return selected, doc.node_meta


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
