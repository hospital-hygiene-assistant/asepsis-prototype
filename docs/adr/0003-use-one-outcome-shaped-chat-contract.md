# ADR 0003: Use One Outcome-Shaped Chat Contract

## Status

Accepted

## Context

The former chat wire repeated search coverage beside answer and error fields.
Consumers had to reconcile combinations that should never coexist, such as an
honest negative with an incomplete search. It also transmitted both structured
answer sections and a second concatenated copy of the same prose.

The Next.js practitioner surface and the backend debug console are the only
consumers. Carrying a compatibility reader would preserve the ambiguity and
create a second contract path without a real third-party adapter.

## Decision

Version 3 is a strict discriminated outcome: `answered`,
`insufficient_evidence`, `search_incomplete`, `retrieval_unavailable`, or
`synthesis_unavailable`.

- Search coverage appears exactly once inside the outcome.
- Structured answer prose exists only for `answered`; consumers construct any
  display string locally.
- Citations name their document identity and selection reason directly.
- Visual evidence is either `exact`, with an immutable generation-scoped PDF
  link and page regions, or `unavailable`, with an explicit reason.
- The practitioner adapter derives qualitative German grounding copy; the wire
  does not transmit a second, potentially contradictory grounding verdict.
- The two consumers move atomically. Version 2 is removed rather than retained
  as a compatibility seam.

The backend fixture producer writes all seven truthful outcome examples to both
repositories and can check them byte-for-byte before handoff.

## Consequences

- Every response says what happened without consumers interpreting field
  presence or HTTP status as clinical search truth.
- Unknown fields and contradictory nested facts fail closed at the frontend
  adapter.
- Adding an outcome is an explicit cross-repository contract change.
- Deployments must update the backend and frontend together.
