"""
Module 3 — Query: Ollama Synthesis
Turns retrieved PageIndex leaf nodes into a cited answer with one Ollama call.
"""

import pageindex as _pi
from pageindex import PageNode

MODULE_INFO = {
    "stage": "query",
    "name": "ollama_synthesis",
    "label": "Ollama Synthesis",
    "description": "Retrieves leaf nodes across all indexed documents and synthesises a clinically-cited answer using Ollama. Two LLM calls: one per document for retrieval, one for synthesis.",
}

SYNTHESIS_PROMPT_TEMPLATE = """\
You are a medical knowledge assistant. Answer the following query using ONLY the
provided source passages. Cite each claim with the source reference in brackets.

Query: QUERY_PLACEHOLDER

Source passages:
PASSAGES_PLACEHOLDER

Instructions:
- Be concise and clinically accurate.
- Cite sources inline using the reference tags provided.
- After your answer, list the sources you used.
- If the passages do not contain enough information to answer fully, say so.
"""


def synthesise(query: str, nodes_by_doc: dict[str, list[PageNode]]) -> str:
    """Compose an answer from retrieved passages, citing each by document and node."""
    passages = [
        f"[{doc_name} / {node.node_id}]\n{node.content or ''}"
        for doc_name, nodes in nodes_by_doc.items()
        for node in nodes
    ]
    if not passages:
        return "No relevant passages found."

    prompt = (
        SYNTHESIS_PROMPT_TEMPLATE
        .replace("QUERY_PLACEHOLDER", query)
        .replace("PASSAGES_PLACEHOLDER", "\n\n".join(passages))
    )
    # Through the configured pool, so the CLI answers from the same instance as
    # the API rather than whatever host the ollama default points at.
    client = _pi.make_client(_pi.OLLAMA_URLS[0])
    response = client.chat(
        model=getattr(_pi, "SYNTHESIS_MODEL", _pi.MODEL),
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"]
