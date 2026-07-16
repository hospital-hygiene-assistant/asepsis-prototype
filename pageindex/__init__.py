"""PageIndex — deterministic heading-tree indexing with model-guided retrieval.

The build is deterministic: markdown headings become the tree, preamble text
becomes synthetic leaves, summaries are heuristic. The model is consulted only
at query time, to prune sections and judge leaves.

    build_generation(documents)   atomically publish one complete library
    retrieve_with_metadata(...) judge one validated document tree

The pieces live alongside and can be imported directly: settings, clients,
question_run, pins, nodes, build, prompts, llm, search.
"""

from paths import KB_DIR, LIBRARY_DIR

from .clients import OLLAMA_URLS, get_activity, make_client, reconfigure_clients
from .nodes import (
    HeadingIdentity,
    PageNode,
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
    retrieve_with_metadata,
)
from .document_index import DocumentIndex
from .question_run import QuestionRun
from .settings import settings

__all__ = [
    "KB_DIR", "LIBRARY_DIR",
    "HeadingIdentity", "PageNode", "heading_identities", "parse_document",
    "AssetProvenance", "NormalizedRegion", "PinValidationError", "PixelBox",
    "ProvenancePin", "SourceSpan", "VisualLocation", "emit_pin",
    "locate_visual_citation", "parse_pin",
    "build_generation", "retrieve_with_metadata", "explain_nonselection",
    "DocumentIndex",
    "make_client", "reconfigure_clients", "get_activity", "OLLAMA_URLS",
    "QuestionRun",
    "settings",
    "LEAF_EVAL_PROMPT", "SECTION_CHECK_PROMPT", "EXPLAIN_PROMPT",
]
