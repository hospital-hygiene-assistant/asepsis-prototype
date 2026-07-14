"""Reading the indexed heading trees."""

import json
from pathlib import Path

from pageindex import _node_from_dict


def count_leaves(nodes) -> int:
    """Leaves under a list of PageNodes."""
    return sum(1 if node.is_leaf else count_leaves(node.children) for node in nodes)


def read_tree(index_file: Path) -> list[dict]:
    """The serialised tree for one indexed document."""
    return json.loads(index_file.read_text(encoding="utf-8"))


def leaf_count(tree: list[dict]) -> int:
    """Leaves in a serialised tree. Both callers were converting it themselves."""
    return count_leaves([_node_from_dict(node) for node in tree])
