"""
Module 2 — Index: PageIndex (Custom)
Deterministic heading-tree indexing with model-guided retrieval via Ollama.

The build is deterministic: markdown headings become the tree, preamble text
becomes synthetic leaves, summaries are heuristic. The model is consulted only
at query time, to prune sections and judge leaves.
"""

from pageindex import build_index, retrieve, retrieve_with_metadata

# The registry loads this module and calls these off it, so the re-export is the
# whole point. Naming them here states that, where a lint suppression only hid
# the complaint — and pyflakes, which ignores suppressions, reported them anyway.
__all__ = ["build_index", "retrieve", "retrieve_with_metadata", "MODULE_INFO"]

MODULE_INFO = {
    "stage": "index",
    "name": "pageindex_custom",
    "label": "PageIndex (Custom)",
    "description": "Heading-hierarchy tree index with Ollama LLM retrieval. Fully deterministic build; LLM only used at query time to select relevant leaf nodes from a TOC.",
}
