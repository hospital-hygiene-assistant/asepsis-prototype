# ADR 0004: Review PDF Assets Before Publication

## Status

Accepted

## Context

PDF layout detection proposes figure and table regions, but those proposals are
not reliable enough to become clinical evidence without inspection. The earlier
ingest path captioned and published crops directly. A mistaken region or guessed
table structure could therefore enter the current Expected library without a
human decision.

The backend already has a local operator console and an immutable Expected
library publisher. A hosted upload workflow remains outside the local trust
model under ADR 0002.

## Decision

The BetterIngest PDF adapter prepares deterministic document text and proposed
asset regions, then stops in a persistent Ingest review. The operator can add,
move, resize, or remove rectangular figure and table regions in the existing
same-origin console.

- Review sessions use opaque identities, strict typed state, optimistic
  revisions, atomic local persistence, and a 24-hour lifetime.
- Source PDFs and crops are copied into the private review directory. The HTTP
  interface never accepts or returns an arbitrary filesystem path.
- Confirmed table crops use the focused Paddle
  `TableRecognitionPipelineV2` on CPU. Structure is accepted verbatim; it is not
  repaired by a language model. A failed recognition requires explicit operator
  acknowledgement before publication.
- Confirmation constructs canonical Markdown and provenance pins and invokes
  the Expected library publisher once for the complete batch. The current
  generation moves only after the new generation reopens successfully.
- Figures and tables are the complete review scope. General PDF text editing,
  background job orchestration, and hosted upload are not added.

## Consequences

- Layout proposals cannot silently become current evidence.
- A crash or browser refresh can resume an unexpired review, while a stale
  browser revision cannot overwrite newer geometry.
- Tests exercise review through its lifecycle interface and publication through
  the existing immutable-library interface.
- Only one active review is supported. Multi-user queues and audit identities
  require a hosted-deployment design.
- Paddle model packages are locked, but their separately downloaded model
  weights are not yet content-digest pinned; this is recorded in the runtime
  model manifest.
