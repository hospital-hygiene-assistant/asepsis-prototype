"""
Module 3 — Query: Ollama Synthesis
Retrieves all relevant PageIndex leaf nodes across all indexed documents,
then synthesises a cited answer via a second Ollama call.
"""
# Delegate to the root implementation
from query import (
    run_query,
    _synthesise,
    _format_node,
    SYNTHESIS_PROMPT_TEMPLATE,
)

MODULE_INFO = {
    "stage": "query",
    "name": "ollama_synthesis",
    "label": "Ollama Synthesis",
    "description": "Retrieves leaf nodes across all indexed documents and synthesises a clinically-cited answer using Ollama. Two LLM calls: one per document for retrieval, one for synthesis.",
}
