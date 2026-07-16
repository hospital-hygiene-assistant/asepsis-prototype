# ADR 0002: Keep The Current Backend Local-Only

## Status

Accepted

## Context

The current backend is operated by a trusted person on one workstation. Its
ingest interface accepts server-local folder paths, and its configuration
interface controls local Ollama processes. These are useful local controls but
are unsafe assumptions for a network-hosted deployment.

ASEPSIS is patient-adjacent. Quietly treating the current interface as
deployment-ready would hide missing access control and operational safeguards.

## Decision

The current backend remains bound to loopback and uses a local trust model.
Authentication, safe file upload and storage, bounded background-job
orchestration, audit controls, and deployment security are deferred together
to a separate hosted-deployment design.

The existing Tauri-named directory and desktop launcher remain compatibility
details for the local debug console. Renaming or removing them is not required
for the Next.js practitioner surface handoff.

## Consequences

- Local ingest and model controls remain simple and useful to developers.
- Documentation must not describe the current backend as safe to expose on a
  network.
- Any hosted deployment must revisit this decision before changing the bind
  address or placing the backend behind a public route.
- This decision does not add authentication or upload machinery that the local
  prototype does not need.
