"""PageIndex — deterministic heading-tree indexing with model-guided retrieval.

The build is deterministic: markdown headings become the tree, preamble text
becomes synthetic leaves, summaries are heuristic. The model is consulted only
at query time, to prune sections and judge leaves.

    build_index(doc)              write one document's tree to index/
    retrieve(doc, query)          the leaves judged relevant
    retrieve_with_metadata(...)   the same, plus why each was chosen

The pieces live alongside and can be imported directly: settings, clients,
run_state, pins, nodes, build, prompts, llm, search, cli.
"""

from paths import INDEX_DIR, KB_DIR

from .clients import OLLAMA_URLS, get_activity, make_client, reconfigure_clients
from .nodes import PageNode, _build_nodes_by_id, _build_parent_map, _make_breadcrumb, _node_from_dict
from .prompts import EXPLAIN_PROMPT, LEAF_EVAL_PROMPT, SECTION_CHECK_PROMPT
from .build import build_index
from .search import explain_nonselection, retrieve, retrieve_with_metadata
from .run_state import RunState
from .settings import settings

__all__ = [
    "INDEX_DIR", "KB_DIR",
    "PageNode",
    "build_index", "retrieve", "retrieve_with_metadata", "explain_nonselection",
    "make_client", "reconfigure_clients", "get_activity", "OLLAMA_URLS",
    "RunState",
    "settings",
    "LEAF_EVAL_PROMPT", "SECTION_CHECK_PROMPT", "EXPLAIN_PROMPT",
    # The API renders breadcrumbs and rehydrates trees from stored JSON.
    "_node_from_dict", "_build_nodes_by_id", "_build_parent_map", "_make_breadcrumb",
]
