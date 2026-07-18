"""Running a retrieval pass over the whole index.

Shared by /api/run and /api/chat so both drive the same live progress state and
return the same shape.
"""

from dataclasses import dataclass
from enum import StrEnum
import re

import pageindex as _pi
from pageindex.question_run import (
    PassageDecision,
    PassageDecisionKind,
    QuestionCancelled,
    QuestionDeadlineExceeded,
)
from pageindex.library import (
    ExpectedLibraryCorrupt,
    ExpectedLibraryNotBuilt,
    ExpectedLibrarySnapshot,
    ExpectedLibraryStore,
    LibraryDocument,
)
from paths import LIBRARY_DIR

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
    decisions: tuple[PassageDecision, ...]
    evidence: tuple[VerifiedEvidence, ...]
    diagnostics: tuple[SearchDiagnostic, ...]

    @classmethod
    def from_verified_evidence(
        cls,
        document: str,
        evidence: tuple[VerifiedEvidence, ...],
    ) -> "DocumentSearch":
        """Build canonical searched facts for deterministic adapters/fixtures."""
        return cls(
            document=document,
            decisions=tuple(
                PassageDecision(
                    item.node.node_id,
                    PassageDecisionKind.PASSAGE_RETRIEVED,
                    item.reason,
                    item.quote,
                    document_id=document,
                )
                for item in evidence
            ),
            evidence=evidence,
            diagnostics=(),
        )

    def __post_init__(self) -> None:
        if not self.document:
            raise ValueError("document search requires a document identity")
        decisions_by_node = {decision.node_id: decision for decision in self.decisions}
        if len(decisions_by_node) != len(self.decisions):
            raise ValueError("document search contains duplicate node decisions")
        if any(decision.document_id != self.document for decision in self.decisions):
            raise ValueError("passage decision belongs to a different document")
        evidence_by_node = {item.node.node_id: item for item in self.evidence}
        if len(evidence_by_node) != len(self.evidence):
            raise ValueError("document search contains duplicate evidence")
        for node_id, item in evidence_by_node.items():
            decision = decisions_by_node.get(node_id)
            if item.document != self.document:
                raise ValueError("verified evidence belongs to a different document")
            if (
                decision is None
                or decision.kind is not PassageDecisionKind.PASSAGE_RETRIEVED
                or decision.quote != item.quote
                or decision.reason != item.reason
            ):
                raise ValueError("verified evidence contradicts its passage decision")
        diagnosed_nodes = {
            item.node_id for item in self.diagnostics if item.node_id is not None
        }
        if any(
            decision.kind is PassageDecisionKind.PASSAGE_RETRIEVED
            and decision.node_id not in evidence_by_node
            and decision.node_id not in diagnosed_nodes
            for decision in self.decisions
        ):
            raise ValueError("retrieved passage is missing evidence or a diagnostic")
        if self.status is DocumentStatus.UNAVAILABLE:
            if self.evidence or self.decisions:
                raise ValueError("unavailable document cannot carry search facts")

    @property
    def status(self) -> DocumentStatus:
        if any(
            diagnostic.code == "document_unavailable"
            and diagnostic.node_id is None
            for diagnostic in self.diagnostics
        ):
            return DocumentStatus.UNAVAILABLE
        if self.diagnostics:
            return DocumentStatus.INCOMPLETE
        return DocumentStatus.SEARCHED


@dataclass(frozen=True)
class LibrarySearchResult:
    query: str
    generation_id: str | None
    documents: tuple[DocumentSearch, ...]
    availability_diagnostics: tuple[SearchDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.generation_id is None and self.documents:
            raise ValueError("unavailable library cannot carry document results")
        if self.generation_id is not None and not self.documents:
            raise ValueError("opened library requires a non-empty document result")
        identities = tuple(document.document for document in self.documents)
        if len(set(identities)) != len(identities):
            raise ValueError("library search contains duplicate documents")

    @property
    def evidence(self) -> tuple[VerifiedEvidence, ...]:
        return tuple(item for document in self.documents for item in document.evidence)

    @property
    def diagnostics(self) -> tuple[SearchDiagnostic, ...]:
        return self.availability_diagnostics + tuple(
            item for document in self.documents for item in document.diagnostics
        )

    @property
    def status(self) -> LibraryStatus:
        if self.generation_id is None or not self.documents or all(
            document.status is DocumentStatus.UNAVAILABLE for document in self.documents
        ):
            return LibraryStatus.UNAVAILABLE
        if self.diagnostics:
            return LibraryStatus.PARTIAL
        return LibraryStatus.COMPLETE

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
            run.set_total(0)
            diagnostic = SearchDiagnostic(
                document="__expected_library__",
                code="expected_library_unavailable",
                message=str(exc),
            )
            return LibrarySearchResult(
                query=query,
                generation_id=None,
                documents=(),
                availability_diagnostics=(diagnostic,),
            )
        return self._search_snapshot(query, run, snapshot, retrieval_model)

    def _search_snapshot(
        self,
        query: str,
        run,
        snapshot: ExpectedLibrarySnapshot,
        retrieval_model: str | None,
    ) -> LibrarySearchResult:
        prepared: list[tuple[LibraryDocument, SearchDiagnostic | None]] = []
        for library_document in snapshot.documents:
            document = library_document.document_id
            try:
                if library_document.index.leaf_count == 0:
                    raise ValueError("document index has no searchable leaves")
            except Exception as exc:
                diagnostic = SearchDiagnostic(
                    document=document,
                    code="document_unavailable",
                    message=str(exc),
                )
                prepared.append((library_document, diagnostic))
                continue
            prepared.append((library_document, None))

        run.set_total(sum(
            document.index.leaf_count
            for document, diagnostic in prepared if diagnostic is None
        ))
        documents: list[DocumentSearch] = []
        for library_document, read_diagnostic in prepared:
            run.checkpoint()
            document = library_document.document_id
            if read_diagnostic is not None:
                assert read_diagnostic is not None
                documents.append(DocumentSearch(
                    document=document,
                    decisions=(),
                    evidence=(),
                    diagnostics=(read_diagnostic,),
                ))
                continue
            try:
                retrieval_options = {"model": retrieval_model}
                if self._instances is not None:
                    retrieval_options["instances"] = self._instances
                retrieval = self._engine.search_document(
                    document,
                    query,
                    run,
                    library_document.index,
                    **retrieval_options,
                )
            except (QuestionCancelled, QuestionDeadlineExceeded):
                raise
            except Exception as exc:
                diagnostic = SearchDiagnostic(
                    document=document,
                    code="document_unavailable",
                    message=str(exc),
                )
                documents.append(
                    DocumentSearch(
                        document=document,
                        decisions=(),
                        evidence=(),
                        diagnostics=(diagnostic,),
                    )
                )
                continue

            evidence: list[VerifiedEvidence] = []
            decisions = tuple(retrieval.decisions)
            diagnostics = [
                SearchDiagnostic(
                    document=document,
                    node_id=decision.node_id,
                    code=decision.code or "unevaluated_node",
                    message=decision.reason or "The node was not evaluated.",
                )
                for decision in decisions
                if decision.kind.audit_status == "error"
            ]
            for decision in decisions:
                if decision.kind is not PassageDecisionKind.PASSAGE_RETRIEVED:
                    continue
                try:
                    node = library_document.index.node(decision.node_id)
                except KeyError:
                    diagnostics.append(SearchDiagnostic(
                        document=document,
                        node_id=decision.node_id,
                        code="unknown_retrieved_node",
                        message="The retrieved decision named no indexed passage.",
                    ))
                    continue
                quote = decision.quote.strip()
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
                    breadcrumb=library_document.index.breadcrumb(node.node_id),
                    reason=decision.reason,
                    quote=quote,
                ))
            node_order = library_document.index.node_order()
            diagnostics.sort(
                key=lambda item: node_order.get(item.node_id or "", len(node_order))
            )
            documents.append(
                DocumentSearch(
                    document=document,
                    decisions=decisions,
                    evidence=tuple(evidence),
                    diagnostics=tuple(diagnostics),
                )
            )

        return LibrarySearchResult(
            query=query,
            generation_id=snapshot.generation_id,
            documents=tuple(documents),
        )
