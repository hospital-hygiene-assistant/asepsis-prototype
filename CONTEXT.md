# ASEPSIS Backend Context

ASEPSIS searches the complete local guideline library and may only present evidence whose quote and provenance are verified.

## Domain Language

- **Expected library**: the complete set of searchable documents in one immutable generation. A generation binds each deterministic index to its canonical Markdown and, when present, its source PDF and provenance assets.
- **Whole-library search**: one query evaluated against a pinned expected library, producing `complete`, `partial`, or `unavailable` coverage.
- **Verified evidence**: a selected leaf whose non-empty quote occurs verbatim in its canonical content.
- **Search diagnostic**: a named document or passage that could not be evaluated. It is technical state, never a negative finding about guideline content.
- **Question answering**: the single workflow that performs whole-library search, synthesises only from verified evidence, validates every structured answer section's citations, and returns a typed outcome.
- **Question run**: the thread-safe lifecycle of one question, atomically owning typed passage decisions, retrieval totals, synthesis phase, cancellation or deadline, and the retained terminal result exposed through polling/debug projections.
- **Document index**: the validated, navigable heading tree for one Expected library document. It owns node lookup, breadcrumbs, leaf counts, serialization, and the debug projection.
- **Chat outcome**: the version 3 wire result for one question: answered, insufficient evidence, incomplete search, unavailable retrieval, or unavailable synthesis. Search coverage occurs exactly once inside it.
- **Honest negative**: `insufficient_evidence`, emitted only after a complete whole-library search found no verified evidence.
- **Incomplete search**: a partial or unavailable search. It must never be restated as an honest negative.
- **Provenance pin**: versioned identity and exact canonical-text spans linking one indexed leaf to its source PDF.
- **Visual citation**: either an exact set of normalized page regions for a unique verbatim quote, or an explicit unavailable reason. No location is inferred.
- **Library generation**: a full-SHA-256, manifest-backed snapshot atomically promoted as the current expected library. Searches and source links retain that identity so a later promotion cannot change their evidence.
- **Search coverage**: the generation identity and complete, partial, or unavailable extent of one whole-library search. It is carried through successful answers and technical failures; it is never inferred from an empty result.
- **Ingest adapter**: one of the real alternate producers of canonical markdown. Indexing, retrieval, and question answering each have one implementation and are not plugin stages.
- **Ingest review**: the expiring, revisioned operator decision over proposed PDF figures and tables. It owns region geometry and table-recognition outcomes but cannot mutate an Expected library generation.
- **Reviewed asset**: an active figure or table region confirmed through Ingest review. A table is publishable only with recognized structure or an explicit operator acknowledgement that structure recognition failed.
- **Review publication**: the atomic promotion that converts every document in one ready Ingest review into canonical Markdown, provenance pins, PDFs, and Reviewed assets, then merges them into a new Expected library generation. A partial batch is never current.
- **Local trust model**: the backend listens on loopback and trusts the person operating the machine. Folder paths, ingest controls, and model-pool controls are not suitable for a hosted deployment without authenticated access and safe upload orchestration.

## Current Architecture

- The Expected library module owns publication, integrity verification, immutable source reads, the Document index, and generation-scoped document links through one interface.
- The Whole-library retrieval module searches one library generation with one frozen retrieval model and Ollama pool snapshot. Its document results own typed decisions and diagnostics; verified evidence is derived from retrieved decisions, while document status, library coverage, and flattened facts are derived and invariant-checked once from those results.
- The Question answering module is the single interface for verified evidence, structured synthesis, citations, and typed answer outcomes.
- The Question run module queues questions through one bounded FIFO executor, records each document-qualified section or passage decision atomically, and retains the terminal result. Search never reconstructs a decision from polling state, polling preserves document-plus-node identity, and the debug adapter alone renders the tree-oriented legacy representation. The practitioner and debug adapters submit, poll, and cancel the same lifecycle.
- The strict version 3 chat adapter renders Question answering outcomes for both the Next.js practitioner surface and the debug console.
- CLI and HTTP adapters share the same Question answering runtime assembly; neither reconstructs retrieval, synthesis, or provenance rules.
- The Ingest review module owns persistent review sessions, geometry validation, focused table recognition, and one atomic call into the Expected library publisher. The operator console is an adapter over that interface; it does not publish staging files itself.
- The current deployment is local-only. Authentication, safe uploads, bounded job orchestration, and deployment hardening belong to a separate hosted-deployment decision.
