"""Authoritative version 3 wire rendering for Question answering outcomes.

Question answering owns which combinations of search coverage, evidence, and
answer prose are truthful.  This module only renders those already-valid
domain outcomes as one discriminated wire shape.
"""

from collections import defaultdict
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .question_answering import AnswerKind, AnswerOutcome


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegionV3(WireModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def stays_within_page(self):
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("visual region must stay within its page")
        return self


class PageRegionsV3(WireModel):
    page: int = Field(gt=0)
    regions: tuple[RegionV3, ...] = Field(min_length=1)


class ExactVisualV3(WireModel):
    status: Literal["exact"] = "exact"
    pdf_href: str
    pages: tuple[PageRegionsV3, ...] = Field(min_length=1)


class UnavailableVisualV3(WireModel):
    status: Literal["unavailable"] = "unavailable"
    reason: str = Field(min_length=1)
    pdf_href: str | None


VisualV3 = Annotated[
    ExactVisualV3 | UnavailableVisualV3,
    Field(discriminator="status"),
]


class CitationV3(WireModel):
    id: str
    number: int = Field(gt=0)
    document_id: str
    node_id: str
    title: str
    breadcrumb: str
    excerpt: str
    quote: str = Field(min_length=1)
    selection_reason: str
    visual: VisualV3


class AnswerV3(WireModel):
    short_answer: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    limitations: str = Field(min_length=1)


class CoverageV3(WireModel):
    status: Literal["complete", "partial", "unavailable"]
    generation_id: str | None
    searched_documents: int = Field(ge=0)
    total_documents: int = Field(ge=0)
    incomplete_checks: int = Field(ge=0)


class AnsweredOutcomeV3(WireModel):
    kind: Literal["answered"] = "answered"
    coverage: CoverageV3
    answer: AnswerV3
    citations: tuple[CitationV3, ...] = Field(min_length=1)


class InsufficientEvidenceOutcomeV3(WireModel):
    kind: Literal["insufficient_evidence"] = "insufficient_evidence"
    coverage: CoverageV3


class SearchIncompleteOutcomeV3(WireModel):
    kind: Literal["search_incomplete"] = "search_incomplete"
    coverage: CoverageV3


class RetrievalUnavailableOutcomeV3(WireModel):
    kind: Literal["retrieval_unavailable"] = "retrieval_unavailable"
    coverage: CoverageV3


class SynthesisUnavailableOutcomeV3(WireModel):
    kind: Literal["synthesis_unavailable"] = "synthesis_unavailable"
    coverage: CoverageV3
    citations: tuple[CitationV3, ...] = Field(min_length=1)


OutcomeV3 = Annotated[
    AnsweredOutcomeV3
    | InsufficientEvidenceOutcomeV3
    | SearchIncompleteOutcomeV3
    | RetrievalUnavailableOutcomeV3
    | SynthesisUnavailableOutcomeV3,
    Field(discriminator="kind"),
]


class ChatResponseV3(WireModel):
    contract_version: Literal[3] = 3
    run_id: str
    query: str
    outcome: OutcomeV3


def _coverage(outcome: AnswerOutcome) -> CoverageV3:
    coverage = outcome.coverage
    return CoverageV3(
        status=coverage.status.value,
        generation_id=coverage.generation_id,
        searched_documents=coverage.searched_documents,
        total_documents=coverage.total_documents,
        incomplete_checks=coverage.incomplete_checks,
    )


def _visual(citation) -> ExactVisualV3 | UnavailableVisualV3:
    visual = citation.visual
    if visual.status != "exact":
        return UnavailableVisualV3(
            reason=visual.reason or "location_unavailable",
            pdf_href=citation.source_href,
        )
    if citation.source_href is None:
        raise ValueError("exact visual citation requires an immutable PDF link")
    grouped: dict[int, list[RegionV3]] = defaultdict(list)
    for region in visual.regions:
        grouped[region.page].append(RegionV3(
            x=region.x,
            y=region.y,
            width=region.width,
            height=region.height,
        ))
    return ExactVisualV3(
        pdf_href=citation.source_href,
        pages=tuple(
            PageRegionsV3(page=page, regions=tuple(regions))
            for page, regions in sorted(grouped.items())
        ),
    )


def _citations(outcome: AnswerOutcome) -> tuple[CitationV3, ...]:
    return tuple(
        CitationV3(
            id=citation.id,
            number=citation.number,
            document_id=citation.document,
            node_id=citation.node_id,
            title=citation.title,
            breadcrumb=citation.breadcrumb,
            excerpt=citation.excerpt,
            quote=citation.quote,
            selection_reason=citation.reason,
            visual=_visual(citation),
        )
        for citation in outcome.citations
    )


def encode_outcome(outcome: AnswerOutcome, *, run_id: str) -> ChatResponseV3:
    """Render one already-truthful Question answering outcome."""
    coverage = _coverage(outcome)
    if outcome.kind is AnswerKind.ANSWERED:
        assert outcome.answer is not None
        rendered = AnsweredOutcomeV3(
            coverage=coverage,
            answer=AnswerV3(
                short_answer=outcome.answer.short_answer,
                recommended_action=outcome.answer.recommended_action,
                rationale=outcome.answer.rationale,
                limitations=outcome.answer.limitations,
            ),
            citations=_citations(outcome),
        )
    elif outcome.kind is AnswerKind.INSUFFICIENT_EVIDENCE:
        rendered = InsufficientEvidenceOutcomeV3(coverage=coverage)
    elif outcome.kind is AnswerKind.SEARCH_INCOMPLETE:
        rendered = SearchIncompleteOutcomeV3(coverage=coverage)
    elif outcome.kind is AnswerKind.RETRIEVAL_UNAVAILABLE:
        rendered = RetrievalUnavailableOutcomeV3(coverage=coverage)
    else:
        rendered = SynthesisUnavailableOutcomeV3(
            coverage=coverage,
            citations=_citations(outcome),
        )
    return ChatResponseV3(
        run_id=run_id,
        query=outcome.search.query,
        outcome=rendered,
    )
