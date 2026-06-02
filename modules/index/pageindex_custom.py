"""
Module 2 — Index: PageIndex (Custom)
Deterministic heading-based tree indexing with LLM-guided retrieval via Ollama.

Build step is fully deterministic: parses markdown heading hierarchy, promotes
preamble text to synthetic leaf nodes, extracts leaf content, generates heuristic
summaries. Retrieval sends the TOC to Ollama and gets nodeIds back.
"""
# Delegate to the root implementation — all logic lives in pageindex.py
from pageindex import (
    PageNode,
    build_index,
    retrieve,
    INDEX_DIR,
    MODEL,
    _parse_headings,
    _build_tree,
    _promote_preambles,
    _populate_content,
    _populate_summaries,
    _flatten_toc,
    _collect_leaves,
    _find_nodes_by_ids,
    _generate_summary,
    _extract_leaf_content,
    _extract_preamble,
    _clean_node_id,
    _node_to_dict,
    _node_from_dict,
)

MODULE_INFO = {
    "stage": "index",
    "name": "pageindex_custom",
    "label": "PageIndex (Custom)",
    "description": "Heading-hierarchy tree index with Ollama LLM retrieval. Fully deterministic build; LLM only used at query time to select relevant leaf nodes from a TOC.",
}
