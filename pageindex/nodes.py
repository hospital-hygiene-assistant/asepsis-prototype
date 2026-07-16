"""The document tree: nodes, how one is built from markdown, and its shapes.

Deterministic throughout — no model is consulted here. Headings become the
hierarchy, text before the first child becomes a synthetic leaf, and summaries
are heuristic.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .pins import (
    PIN_FENCE_OPEN,
    PinValidationError,
    ProvenancePin,
    _strip_pins,
    parse_pin,
    pin_from_dict,
)

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
    pin: Optional[ProvenancePin] = None

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



@dataclass(frozen=True)
class HeadingIdentity:
    """The stable identity PageIndex assigns to one markdown heading."""

    line_idx: int
    level: int
    title: str
    node_id: str


def heading_identities(text: str) -> list[HeadingIdentity]:
    """
    Return [(line_idx, level, title, node_id), ...] for every heading in text.
    Duplicate heading titles are deduplicated with a -2, -3, ... suffix.
    """
    lines = text.split("\n")
    seen: dict[str, int] = {}
    headings: list[HeadingIdentity] = []

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
        headings.append(HeadingIdentity(i, level, title, node_id))

    return headings


def _parse_headings(text: str) -> list[tuple[int, int, str, str]]:
    """Compatibility shape for internal callers; use heading_identities."""
    return [
        (heading.line_idx, heading.level, heading.title, heading.node_id)
        for heading in heading_identities(text)
    ]


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

def _attach_pins(
    nodes: list[PageNode], lines: list[str], document: str | None = None
) -> None:
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
                pin = parse_pin("\n".join(body))
                if pin.node_id != node.node_id:
                    raise PinValidationError(
                        f"pin node identity {pin.node_id!r} does not match "
                        f"heading identity {node.node_id!r}"
                    )
                if document is not None and pin.document != document:
                    raise PinValidationError(
                        f"pin document {pin.document!r} does not match "
                        f"indexed document {document!r}"
                    )
                node.pin = pin
                break
            if stripped and not stripped.startswith("```"):
                break  # real content before any pin → this heading has none
            i += 1
        _attach_pins(node.children, lines, document)


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
        "pin": node.pin.to_dict() if node.pin is not None else None,
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
        pin=pin_from_dict(d["pin"]) if d.get("pin") is not None else None,
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


def parse_document(text: str, document: str | None = None) -> list[PageNode]:
    """Build the complete deterministic PageIndex tree from markdown."""
    lines = text.split("\n")
    nodes = _build_tree(_parse_headings(text))
    _attach_pins(nodes, lines, document)
    _promote_preambles(nodes, lines)
    _populate_content(nodes, lines)
    _populate_summaries(nodes)
    return nodes


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
