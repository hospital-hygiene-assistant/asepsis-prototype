# ASEPSIS Backend Context

ASEPSIS searches the complete local guideline library and may only present evidence whose quote and provenance are verified.

## Domain Language

- **Expected library**: the complete set of searchable documents in one immutable generation. A generation binds each deterministic index to its canonical Markdown and, when present, its source PDF and provenance assets.
- **Whole-library search**: one query evaluated against a pinned expected library, producing `complete`, `partial`, or `unavailable` coverage.
- **Verified evidence**: a selected leaf whose non-empty quote occurs verbatim in its canonical content.
- **Search diagnostic**: a named document or passage that could not be evaluated. It is technical state, never a negative finding about guideline content.
- **Question answering**: the single workflow that performs whole-library search, synthesises only from verified evidence, validates every structured answer section's citations, and returns a typed outcome.
- **Honest negative**: `insufficient_evidence`, emitted only after a complete whole-library search found no verified evidence.
- **Incomplete search**: a partial or unavailable search. It must never be restated as an honest negative.
- **Provenance pin**: versioned identity and exact canonical-text spans linking one indexed leaf to its source PDF.
- **Visual citation**: either an exact set of normalized page regions for a unique verbatim quote, or an explicit unavailable reason. No location is inferred.
- **Library generation**: a full-SHA-256, manifest-backed snapshot atomically promoted as the current expected library. Searches and source links retain that identity so a later promotion cannot change their evidence.
- **Search coverage**: the generation identity and complete, partial, or unavailable extent of one whole-library search. It is carried through successful answers and technical failures; it is never inferred from an empty result.
- **Ingest adapter**: one of the real alternate producers of canonical markdown. Indexing, retrieval, and question answering each have one implementation and are not plugin stages.

## Current Architecture

- The Expected library module owns publication, integrity verification, immutable source lookup, and generation-scoped document links through one interface.
- The Whole-library retrieval module searches one library generation with one frozen retrieval model and Ollama pool snapshot.
- The Question answering module is the single interface for verified evidence, structured synthesis, citations, and typed answer outcomes.
- CLI and HTTP adapters share the same Question answering runtime assembly; neither reconstructs retrieval, synthesis, or provenance rules.
