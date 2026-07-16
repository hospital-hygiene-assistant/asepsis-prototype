"""Building a document's heading tree.

Deterministic: no model is consulted. Headings become the hierarchy, text before
the first child becomes a synthetic leaf, summaries are heuristic, and pins are
lifted out of the markdown onto the node.
"""

from paths import KB_DIR, LIBRARY_DIR, SOURCES_MANIFEST

from .library import (
    ExpectedLibrarySnapshot,
    ExpectedLibraryStore,
    LibraryCandidate,
    read_source_candidates,
)


def build_generation(doc_names: list[str]) -> ExpectedLibrarySnapshot:
    """Build and atomically promote one complete immutable corpus index."""
    documents: dict[str, LibraryCandidate] = {}
    source_candidates = read_source_candidates(SOURCES_MANIFEST)
    for doc_name in sorted(doc_names):
        print(f"  Indexing {doc_name}...")
        document_path = KB_DIR / f"{doc_name}.md"
        if not document_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
        documents[doc_name] = LibraryCandidate(
            canonical_markdown=document_path.read_text(encoding="utf-8"),
            source=source_candidates.get(doc_name),
        )
        print(f"    staged {doc_name}")
    snapshot = ExpectedLibraryStore(LIBRARY_DIR).publish(documents)
    print(f"    → promoted Expected library {snapshot.generation_id}")
    return snapshot
