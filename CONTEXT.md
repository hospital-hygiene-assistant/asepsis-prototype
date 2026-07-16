# ASEPSIS Backend Context

ASEPSIS searches the complete local guideline library and may only present evidence whose quote and provenance are verified.

## Domain Language

- **Expected library**: the complete set of documents in one immutable index generation.
- **Whole-library search**: one query evaluated against a pinned expected library, producing `complete`, `partial`, or `unavailable` coverage.
- **Verified evidence**: a selected leaf whose non-empty quote occurs verbatim in its canonical content.
- **Search diagnostic**: a named document or passage that could not be evaluated. It is technical state, never a negative finding about guideline content.
- **Question answering**: the single workflow that performs whole-library search, synthesises only from verified evidence, validates every structured answer section's citations, and returns a typed outcome.
- **Honest negative**: `insufficient_evidence`, emitted only after a complete whole-library search found no verified evidence.
- **Incomplete search**: a partial or unavailable search. It must never be restated as an honest negative.
- **Provenance pin**: versioned identity and exact canonical-text spans linking one indexed leaf to its source PDF.
- **Visual citation**: either an exact set of normalized page regions for a unique verbatim quote, or an explicit unavailable reason. No location is inferred.
- **Index generation**: an immutable, manifest-backed snapshot atomically promoted as the current expected library and pinned for the duration of a search.
- **Ingest adapter**: one of the real alternate producers of canonical markdown. Indexing, retrieval, and question answering each have one implementation and are not plugin stages.
