"""Prompts for the retrieval and explanation calls."""

LEAF_EVAL_PROMPT = """\
Decide if the section below directly answers the query or an explicit part of it. Read the full content carefully.

Query: QUERY_PLACEHOLDER

Document: DOCNAME_PLACEHOLDER
Section path: BREADCRUMB_PLACEHOLDER
Parent section: PARENT_SUMMARY_PLACEHOLDER

--- SECTION CONTENT ---
CONTENT_PLACEHOLDER
--- END ---

Rules:
- Answer YES only if the content directly and specifically addresses the query.
- If the query asks multiple questions, answer YES when the content directly and specifically answers at least one of them; one section need not answer every part.
- A section that is tangentially related or only mentions the topic in passing is NOT relevant.
- If YES, the quote must be copied character-for-character from the content above.

Output exactly one of:
  {"relevant": true,  "reason": "1-2 sentences: what in this content answers the query", "quote": "1-2 verbatim sentences from the content that best answer the query"}
  {"relevant": false}

Output only the JSON object. No other text.
"""

SECTION_CHECK_PROMPT = """\
Could the document section below contain content that directly answers the query?

Query: QUERY_PLACEHOLDER
Section: BREADCRUMB_PLACEHOLDER

This section and all of its nested subsections / headings:
DESCENDANTS_PLACEHOLDER

Rules:
- Answer in JSON format.
- If ANY of the headings above might cover content that answers the query, answer:
  {"relevant": true, "reason": "1-2 sentences explaining why this section was selected"}
  The "reason" field is STRICTLY REQUIRED when "relevant" is true.
- If you are certain that none of these headings could contain a direct answer, answer:
  {"relevant": false}

Output only the JSON object. No other text.
"""

# On-demand explanation for a section that was NOT selected (rejected or pruned).
# Anchored to the actual content so the negative is verifiable, not confabulated:
# the model must describe what the section factually covers, then judge fit.
EXPLAIN_PROMPT = """\
A section of a document was not selected as answering a query. Read the section
content below and explain, factually, what it actually covers.

Query: QUERY_PLACEHOLDER
Document: DOCNAME_PLACEHOLDER
Section path: BREADCRUMB_PLACEHOLDER

--- SECTION CONTENT ---
CONTENT_PLACEHOLDER
--- END ---

First describe the section's actual topic, grounded strictly in the content above.
Then judge whether it directly answers the query.

Output exactly this JSON object:
  {"topic": "one factual sentence describing what this section actually covers", "addresses_query": false, "reason": "one sentence contrasting the section's topic with what the query asks for"}

Set "addresses_query" to true ONLY if, on re-reading, the content does directly answer the query.
Output only the JSON object. No other text.
"""
