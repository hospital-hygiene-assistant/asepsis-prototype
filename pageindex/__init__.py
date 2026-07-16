"""PageIndex — deterministic heading-tree indexing with model-guided retrieval.

The build is deterministic: markdown headings become the tree, preamble text
becomes synthetic leaves, summaries are heuristic. The model is consulted only
at query time, to prune sections and judge leaves.

    build_generation(documents)   atomically publish one complete library
    retrieve_with_metadata_from_path(...) judge one pinned document tree

The pieces live alongside and can be imported directly: settings, clients,
run_state, pins, nodes, build, prompts, llm, search.
"""

from paths import KB_DIR, LIBRARY_DIR

from .clients import OLLAMA_URLS, get_activity, make_client, reconfigure_clients
from .nodes import (
    HeadingIdentity,
    PageNode,
    _build_nodes_by_id,
    _build_parent_map,
    _make_breadcrumb,
    _node_from_dict,
    heading_identities,
    parse_document,
)
from .pins import (
    AssetProvenance,
    NormalizedRegion,
    PinValidationError,
    PixelBox,
    ProvenancePin,
    SourceSpan,
    VisualLocation,
    emit_pin,
    locate_visual_citation,
    parse_pin,
)
from .prompts import EXPLAIN_PROMPT, LEAF_EVAL_PROMPT, SECTION_CHECK_PROMPT
from .build import build_generation
from .search import (
    explain_nonselection,
    retrieve_with_metadata_from_path,
)
from .run_state import RunState
from .settings import settings

__all__ = [
    "KB_DIR", "LIBRARY_DIR",
    "HeadingIdentity", "PageNode", "heading_identities", "parse_document",
    "AssetProvenance", "NormalizedRegion", "PinValidationError", "PixelBox",
    "ProvenancePin", "SourceSpan", "VisualLocation", "emit_pin",
    "locate_visual_citation", "parse_pin",
    "build_generation", "retrieve_with_metadata_from_path", "explain_nonselection",
    "make_client", "reconfigure_clients", "get_activity", "OLLAMA_URLS",
    "RunState",
    "settings",
    "LEAF_EVAL_PROMPT", "SECTION_CHECK_PROMPT", "EXPLAIN_PROMPT",
    # The API renders breadcrumbs and rehydrates trees from stored JSON.
    "_node_from_dict", "_build_nodes_by_id", "_build_parent_map", "_make_breadcrumb",
]
