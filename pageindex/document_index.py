"""A validated, navigable PageIndex document.

This is the single read model for a published document tree. Callers receive
typed nodes and domain operations instead of reopening JSON files or importing
PageIndex's private traversal helpers.
"""

import json
from collections.abc import Sequence
from types import MappingProxyType

from .nodes import (
    PageNode,
    _build_nodes_by_id,
    _build_parent_map,
    _collect_leaves,
    _make_breadcrumb,
)
from .serialization import deserialize_document, serialize_document


class DocumentIndex:
    """One validated document tree with its navigation operations."""

    def __init__(self, nodes: Sequence[PageNode]) -> None:
        # Round-tripping once applies the same validation and gives this read
        # model ownership of the graph rather than retaining a caller's list.
        self._payload = serialize_document(nodes)
        self._nodes = deserialize_document(self._payload)
        mutable_nodes_by_id = _build_nodes_by_id(list(self._nodes))
        if len(mutable_nodes_by_id) != sum(1 for _ in self._walk(self._nodes)):
            raise ValueError("document tree contains duplicate node identities")
        self._nodes_by_id = MappingProxyType(mutable_nodes_by_id)
        self._parent_map = MappingProxyType(_build_parent_map(list(self._nodes)))
        self._leaves = tuple(_collect_leaves(list(self._nodes)))

    @classmethod
    def from_nodes(cls, nodes: Sequence[PageNode]) -> "DocumentIndex":
        return cls(nodes)

    @classmethod
    def from_serialized(cls, payload: str) -> "DocumentIndex":
        return cls(deserialize_document(payload))

    @property
    def nodes(self) -> tuple[PageNode, ...]:
        return self._nodes

    @property
    def leaves(self) -> tuple[PageNode, ...]:
        return self._leaves

    @property
    def leaf_count(self) -> int:
        return len(self._leaves)

    @property
    def debug_tree(self) -> list[dict]:
        """A fresh JSON projection for the retrieval explorer."""
        return json.loads(self._payload)

    def serialize(self) -> str:
        return self._payload

    def node(self, node_id: str) -> PageNode:
        try:
            return self._nodes_by_id[node_id]
        except KeyError as exc:
            raise KeyError(node_id) from exc

    def breadcrumb(self, node_id: str) -> str:
        if node_id not in self._nodes_by_id:
            raise KeyError(node_id)
        return _make_breadcrumb(node_id, self._parent_map, self._nodes_by_id)

    def node_order(self) -> dict[str, int]:
        return {
            node.node_id: position
            for position, node in enumerate(self._walk(self._nodes))
        }

    @classmethod
    def _walk(cls, nodes: Sequence[PageNode]):
        for node in nodes:
            yield node
            yield from cls._walk(node.children)
