"""
Module 2 — Index: PageIndex (Custom)
Deterministic heading-based tree indexing with LLM-guided retrieval via Ollama.

Build step is fully deterministic: parses markdown heading hierarchy, promotes
preamble text to synthetic leaf nodes, extracts leaf content, generates heuristic
summaries. Retrieval sends the TOC to Ollama and gets nodeIds back.
"""
# Delegate to the root implementation — all logic lives in pageindex.py
#
# prune_document + evaluate_ranked are the TWO-PHASE API, and re-exporting them
# is not optional decoration: the server checks for them with hasattr and falls
# back to the single-document `retrieve_with_metadata` when they are missing.
# They were absent, so every query in the app took the fallback — which drops
# `selected_answers` entirely and budgets per document instead of across the
# corpus. The multiple-choice pre-filter therefore never filtered anything in
# the running app, however correct the code behind it was.
from pageindex import (
    PageNode,
    RunContext,
    build_index,
    prune_document,
    evaluate_ranked,
    retrieve,
    retrieve_with_metadata,
    INDEX_DIR,
    MODEL,
    _parse_headings,
    _build_tree,
    _promote_preambles,
    _populate_content,
    _populate_summaries,
    _collect_leaves,
    _generate_summary,
    _extract_leaf_content,
    _extract_preamble,
    _node_to_dict,
    _node_from_dict,
)

MODULE_INFO = {
    "stage": "index",
    "name": "pageindex_custom",
    "label": "PageIndex (Custom)",
    "description": "Heading-hierarchy tree index with Ollama LLM retrieval. Fully deterministic build; LLM only used at query time to select relevant leaf nodes from a TOC.",
}
