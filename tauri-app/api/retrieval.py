"""Running a retrieval pass over the whole index.

Shared by /api/run and /api/chat so both drive the same live progress state and
return the same shape.
"""

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any

import pageindex as _pi
from pageindex.library import (
    ExpectedLibraryCorrupt,
    ExpectedLibraryNotBuilt,
    ExpectedLibrarySnapshot,
    ExpectedLibraryStore,
    LibraryDocument,
)
from paths import LIBRARY_DIR

from .trees import leaf_count, read_tree


class LibraryStatus(StrEnum):
    """How completely the expected document library was searched."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class DocumentStatus(StrEnum):
    """Whether one expected document received a valid retrieval pass."""

    SEARCHED = "searched"
    INCOMPLETE = "incomplete"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class SearchDiagnostic:
    document: str
    code: str
    message: str
    node_id: str | None = None


@dataclass(frozen=True)
class SearchCoverage:
    """Truthful extent of one Whole-library search."""

    status: LibraryStatus
    generation_id: str | None
    searched_documents: int
    total_documents: int
    incomplete_checks: int

    def __post_init__(self) -> None:
        if (
            self.generation_id is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.generation_id) is None
        ):
            raise ValueError(
                "generation_id must be 64 lowercase hexadecimal characters"
            )
        if self.total_documents < 0 or not 0 <= self.searched_documents <= self.total_documents:
            raise ValueError("search coverage document counts are invalid")
        if self.incomplete_checks < 0:
            raise ValueError("search coverage incomplete checks cannot be negative")
        if self.status is LibraryStatus.COMPLETE:
            if self.generation_id is None:
                raise ValueError("complete coverage requires a generation identity")
            if self.total_documents == 0:
                raise ValueError("complete coverage requires a non-empty expected library")
            if (
                self.searched_documents != self.total_documents
                or self.incomplete_checks != 0
            ):
                raise ValueError(
                    "complete coverage requires every document and no incomplete checks"
                )
        elif self.status is LibraryStatus.PARTIAL:
            if self.generation_id is None:
                raise ValueError("partial coverage requires a generation identity")
            if self.total_documents == 0 or self.searched_documents == 0:
                raise ValueError(
                    "partial coverage requires a non-empty expected library and "
                    "at least one searched document"
                )
            if (
                self.searched_documents == self.total_documents
                and self.incomplete_checks == 0
            ):
                raise ValueError("partial coverage requires incomplete facts")
        elif self.searched_documents != 0:
            raise ValueError("unavailable coverage cannot contain searched documents")


@dataclass(frozen=True)
class VerifiedEvidence:
    document: str
    node: _pi.PageNode
    breadcrumb: str
    reason: str
    quote: str


@dataclass(frozen=True)
class DocumentSearch:
    document: str
    status: DocumentStatus
    tree: tuple[dict, ...]
    node_meta: dict[str, dict]
    evidence: tuple[VerifiedEvidence, ...]
    diagnostics: tuple[SearchDiagnostic, ...]


@dataclass(frozen=True)
class LibrarySearchResult:
    query: str
    generation_id: str | None
    status: LibraryStatus
    documents: tuple[DocumentSearch, ...]
    evidence: tuple[VerifiedEvidence, ...]
    diagnostics: tuple[SearchDiagnostic, ...]

    @property
    def coverage(self) -> SearchCoverage:
        return SearchCoverage(
            status=self.status,
            generation_id=self.generation_id,
            searched_documents=sum(
                document.status is not DocumentStatus.UNAVAILABLE
                for document in self.documents
            ),
            total_documents=len(self.documents),
            incomplete_checks=len(self.diagnostics),
        )

    def document(self, name: str) -> DocumentSearch | None:
        return next((document for document in self.documents if document.document == name), None)

    def to_debug_results(self) -> dict[str, dict]:
        """Adapter for the retrieval explorer's tree-oriented representation."""
        results: dict[str, dict] = {}
        for document in self.documents:
            node_meta = {
                node_id: dict(meta) for node_id, meta in document.node_meta.items()
            }
            for position, diagnostic in enumerate(document.diagnostics):
                node_id = diagnostic.node_id or f"__document__:{position}"
                node_meta[node_id] = {
                    "status": "error",
                    "reason": diagnostic.message,
                    "quote": "",
                    "code": diagnostic.code,
                }
            results[document.document] = {
                "tree": list(document.tree),
                "retrieved_ids": [item.node.node_id for item in document.evidence],
                "node_meta": node_meta,
                "nodes": [_legacy_node(item) for item in document.evidence],
            }
        return results


def _legacy_node(evidence: VerifiedEvidence) -> dict[str, Any]:
    node = evidence.node
    pin = node.pin.to_dict() if node.pin is not None else None
    return {
        "node_id": node.node_id,
        "title": node.title,
        "content": node.content or "",
        "synthetic": node.synthetic,
        "heading_level": node.heading_level,
        "summary": node.summary,
        "pin": pin,
        "breadcrumb": evidence.breadcrumb,
        "reason": evidence.reason,
        "quote": evidence.quote,
    }


def _document_node_order(tree: tuple[dict, ...]) -> dict[str, int]:
    order: dict[str, int] = {}

    def visit(nodes: list[dict] | tuple[dict, ...]) -> None:
        for node in nodes:
            order[node["nodeId"]] = len(order)
            visit(node.get("children", []))

    visit(tree)
    return order


class WholeLibraryRetrieval:
    """Search the expected library and expose one truthful coverage result."""

    def __init__(
        self,
        engine,
        *,
        library_store: ExpectedLibraryStore | None = None,
        retrieval_model: str | None = None,
        instances: tuple | None = None,
    ) -> None:
        self._engine = engine
        self._library_store = library_store or ExpectedLibraryStore(LIBRARY_DIR)
        self._retrieval_model = retrieval_model
        self._instances = instances

    def search(self, query: str, run) -> LibrarySearchResult:
        engine_settings = getattr(self._engine, "settings", None)
        retrieval_model = self._retrieval_model or (
            str(engine_settings.model) if engine_settings is not None else None
        )
        try:
            snapshot = self._library_store.open_current()
        except (ExpectedLibraryNotBuilt, ExpectedLibraryCorrupt) as exc:
            run.state.start(0)
            diagnostic = SearchDiagnostic(
                document="__expected_library__",
                code="expected_library_unavailable",
                message=str(exc),
            )
            return LibrarySearchResult(
                query=query,
                generation_id=None,
                status=LibraryStatus.UNAVAILABLE,
                documents=(),
                evidence=(),
                diagnostics=(diagnostic,),
            )
        return self._search_snapshot(query, run, snapshot, retrieval_model)

    def _search_snapshot(
        self,
        query: str,
        run,
        snapshot: ExpectedLibrarySnapshot,
        retrieval_model: str | None,
    ) -> LibrarySearchResult:
        prepared: list[
            tuple[LibraryDocument, tuple[dict, ...] | None, SearchDiagnostic | None]
        ] = []
        for library_document in snapshot.documents:
            document = library_document.document_id
            try:
                tree = tuple(read_tree(library_document.index_path))
                if leaf_count(tree) == 0:
                    raise ValueError("document index has no searchable leaves")
            except Exception as exc:
                diagnostic = SearchDiagnostic(
                    document=document,
                    code="document_unavailable",
                    message=str(exc),
                )
                prepared.append((library_document, None, diagnostic))
                continue
            prepared.append((library_document, tree, None))

        run.state.start(sum(
            leaf_count(tree) for _, tree, _ in prepared if tree is not None
        ))
        documents: list[DocumentSearch] = []
        for library_document, tree, read_diagnostic in prepared:
            document = library_document.document_id
            if tree is None:
                assert read_diagnostic is not None
                documents.append(DocumentSearch(
                    document=document,
                    status=DocumentStatus.UNAVAILABLE,
                    tree=(),
                    node_meta={},
                    evidence=(),
                    diagnostics=(read_diagnostic,),
                ))
                continue
            try:
                retrieval_options = {"model": retrieval_model}
                if self._instances is not None:
                    retrieval_options["instances"] = self._instances
                nodes, node_meta = self._engine.retrieve_with_metadata_from_path(
                    document,
                    query,
                    run.state,
                    library_document.index_path,
                    **retrieval_options,
                )
            except Exception as exc:
                diagnostic = SearchDiagnostic(
                    document=document,
                    code="document_unavailable",
                    message=str(exc),
                )
                documents.append(
                    DocumentSearch(
                        document=document,
                        status=DocumentStatus.UNAVAILABLE,
                        tree=tree,
                        node_meta={},
                        evidence=(),
                        diagnostics=(diagnostic,),
                    )
                )
                continue

            evidence: list[VerifiedEvidence] = []
            tree_nodes = [_pi._node_from_dict(item) for item in tree]
            nodes_by_id = _pi._build_nodes_by_id(tree_nodes)
            parent_map = _pi._build_parent_map(tree_nodes)
            diagnostics = [
                SearchDiagnostic(
                    document=document,
                    node_id=node_id,
                    code=str(meta.get("code") or "unevaluated_node"),
                    message=str(meta.get("reason") or "The node was not evaluated."),
                )
                for node_id, meta in node_meta.items()
                if meta.get("status") == "error"
            ]
            for node in nodes:
                meta = node_meta.get(node.node_id) or {}
                quote = str(meta.get("quote") or "").strip()
                if not quote or quote not in (node.content or ""):
                    diagnostics.append(
                        SearchDiagnostic(
                            document=document,
                            node_id=node.node_id,
                            code="invalid_quote",
                            message="The selected passage did not contain its non-empty quote.",
                        )
                    )
                    continue
                evidence.append(VerifiedEvidence(
                    document=document,
                    node=node,
                    breadcrumb=_pi._make_breadcrumb(
                        node.node_id, parent_map, nodes_by_id
                    ),
                    reason=str(meta.get("reason") or ""),
                    quote=quote,
                ))
            node_order = _document_node_order(tree)
            diagnostics.sort(
                key=lambda item: node_order.get(item.node_id or "", len(node_order))
            )
            documents.append(
                DocumentSearch(
                    document=document,
                    status=(
                        DocumentStatus.INCOMPLETE
                        if diagnostics
                        else DocumentStatus.SEARCHED
                    ),
                    tree=tree,
                    node_meta=node_meta,
                    evidence=tuple(evidence),
                    diagnostics=tuple(diagnostics),
                )
            )

        all_evidence = tuple(item for document in documents for item in document.evidence)
        all_diagnostics = tuple(
            item for document in documents for item in document.diagnostics
        )
        if not documents or all(
            document.status is DocumentStatus.UNAVAILABLE for document in documents
        ):
            status = LibraryStatus.UNAVAILABLE
        elif all_diagnostics:
            status = LibraryStatus.PARTIAL
        else:
            status = LibraryStatus.COMPLETE
        return LibrarySearchResult(
            query=query,
            generation_id=snapshot.generation_id,
            status=status,
            documents=tuple(documents),
            evidence=all_evidence,
            diagnostics=all_diagnostics,
        )
