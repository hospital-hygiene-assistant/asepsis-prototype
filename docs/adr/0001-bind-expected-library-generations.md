# ADR 0001: Bind Searchable Documents To Immutable Library Generations

## Status

Accepted

## Context

The former corpus layout kept deterministic indexes, canonical Markdown, source PDFs, and provenance assets behind separate shallow modules. An index generation could remain current while ingest replaced the source manifest or files it referenced. A citation could therefore name a passage from one document state and open a PDF from another.

ASEPSIS is patient-adjacent. A visible source error is safer than a plausible but unrelated document page.

## Decision

The Expected library module publishes one immutable generation containing the complete searchable document set. Each document record binds:

- its deterministic heading index;
- its canonical Markdown;
- its source PDF and OCR scale, when present;
- its declared provenance assets, when present.

Generation identity is the full SHA-256 of canonical manifest facts. Objects are stored by content digest and verified when a generation is opened. The current pointer is promoted atomically only after the complete generation can be reopened successfully.

Whole-library retrieval opens exactly one generation. Citations, page geometry, document readers, PDFs, and assets retain that generation identity in their links. A rebuild is required after this change; there is no legacy index reader.

Published generations and objects are retained. Garbage collection, leases, and retention policy require a separate decision and are not implied by this ADR.

## Consequences

- A citation cannot silently drift to a newly ingested PDF or Markdown file.
- Corruption or a missing object makes the Expected library unavailable instead of producing a document claim.
- Tests can construct a generation through the public publisher and exercise retrieval through the same interface used at runtime.
- The `library/` directory replaces the former runtime `index/` layout and consumes additional disk space by retaining immutable generations.
