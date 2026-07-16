"""The one validated representation of a PageIndex document tree."""

import json
from collections.abc import Sequence

from .nodes import PageNode, _node_from_dict, _node_to_dict


def serialize_document(nodes: Sequence[PageNode]) -> str:
    """Serialize a typed document tree for an Expected library generation."""
    if not nodes:
        raise ValueError("a document tree must not be empty")
    if not any(
        node.is_leaf and bool((node.content or "").strip())
        for node in _walk(nodes)
    ):
        raise ValueError("a document tree must contain a searchable leaf")
    return json.dumps(
        [_node_to_dict(node) for node in nodes],
        indent=2,
        ensure_ascii=False,
    )


def deserialize_document(payload: str) -> tuple[PageNode, ...]:
    """Restore a document through the real PageNode deserializer."""
    try:
        raw = json.loads(payload)
        if not isinstance(raw, list) or not raw:
            raise ValueError("a document tree must be a non-empty list")
        nodes = tuple(_node_from_dict(item) for item in raw)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid serialized document tree") from exc
    if not any(
        node.is_leaf and bool((node.content or "").strip())
        for node in _walk(nodes)
    ):
        raise ValueError("a document tree must contain a searchable leaf")
    return nodes


def _walk(nodes: Sequence[PageNode]):
    for node in nodes:
        yield node
        yield from _walk(node.children)
