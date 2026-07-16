"""Authoritative version 2 wire producer for Question answering outcomes."""

from collections import defaultdict
import re
from typing import Annotated, Literal, Self
from urllib.parse import quote, unquote

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .question_answering import AnswerKind, AnswerOutcome
from .retrieval import LibraryStatus

GENERATION_ID_RE = re.compile(r"[0-9a-f]{64}")


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegionV2(WireModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def stays_on_page(self) -> Self:
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("visual region must stay within its page")
        return self


class PageRegionsV2(WireModel):
    page: int = Field(gt=0)
    regions: tuple[RegionV2, ...] = Field(min_length=1)


class ExactVisualV2(WireModel):
    status: Literal["exact"] = "exact"
    pages: tuple[PageRegionsV2, ...] = Field(min_length=1)


class UnavailableVisualV2(WireModel):
    status: Literal["unavailable"] = "unavailable"
    reason: str = Field(min_length=1)


VisualV2 = Annotated[
    ExactVisualV2 | UnavailableVisualV2,
    Field(discriminator="status"),
]


class SourceV2(WireModel):
    id: str
    number: int = Field(gt=0)
    document: str
    node_id: str
    title: str
    breadcrumb: str
    excerpt: str
    quote: str = Field(min_length=1)
    source_href: str | None
    visual: VisualV2

    @model_validator(mode="after")
    def exact_visual_has_source(self) -> Self:
        if self.source_href is not None:
            match = re.fullmatch(
                r"/api/library/([^/]+)/documents/([^/]+)/pdf",
                self.source_href,
            )
            decoded_document = unquote(match.group(2)) if match else ""
            invalid_document = (
                not decoded_document
                or decoded_document in (".", "..")
                or "/" in decoded_document
                or "\\" in decoded_document
                or any(ord(character) < 32 or ord(character) == 127
                       for character in decoded_document)
                or decoded_document != self.document
            )
            if (
                match is None
                or any(
                    not segment
                    or quote(unquote(segment), safe="") != segment
                    for segment in match.groups()
                )
                or GENERATION_ID_RE.fullmatch(match.group(1)) is None
                or invalid_document
            ):
                raise ValueError(
                    "source_href must be the canonical generation-scoped route"
                )
        if self.visual.status == "exact" and self.source_href is None:
            raise ValueError("exact visual requires source_href")
        return self


class AnswerV2(WireModel):
    content: str
    short_answer: str
    recommended_action: str
    rationale: str
    limitations: str


class GroundingV2(WireModel):
    status: Literal[
        "grounded",
        "partially_grounded",
        "insufficient_evidence",
        "search_incomplete",
    ]
    generation_id: str
    searched_documents: int = Field(ge=0)
    total_documents: int = Field(ge=0)
    incomplete_checks: int = Field(ge=0)
    sources: tuple[SourceV2, ...]

    @model_validator(mode="after")
    def citation_numbers_are_contiguous(self) -> Self:
        if GENERATION_ID_RE.fullmatch(self.generation_id) is None:
            raise ValueError(
                "generation_id must be 64 lowercase hexadecimal characters"
            )
        numbers = [source.number for source in self.sources]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("source citation numbers must be contiguous")
        complete = (
            self.total_documents > 0
            and self.searched_documents == self.total_documents
            and self.incomplete_checks == 0
        )
        partial = (
            self.total_documents > 0
            and 0 < self.searched_documents <= self.total_documents
            and (
                self.searched_documents < self.total_documents
                or self.incomplete_checks > 0
            )
        )
        has_sources = bool(self.sources)
        valid = (
            self.status == "grounded" and complete and has_sources
        ) or (
            self.status == "partially_grounded" and partial and has_sources
        ) or (
            self.status == "insufficient_evidence" and complete and not has_sources
        ) or (
            self.status == "search_incomplete" and partial and not has_sources
        )
        if not valid:
            raise ValueError("grounding status contradicts coverage or sources")
        return self


class ChatSuccessV2(WireModel):
    contract_version: Literal[2] = 2
    run_id: str
    query: str
    answer: AnswerV2
    grounding: GroundingV2

    @model_validator(mode="after")
    def grounded_sections_are_citation_backed(self) -> Self:
        for source in self.grounding.sources:
            if source.source_href is None:
                continue
            href_generation = source.source_href.split("/", 4)[3]
            if href_generation != self.grounding.generation_id:
                raise ValueError("source_href must name the grounding generation")
        if self.grounding.status not in ("grounded", "partially_grounded"):
            if any((
                self.answer.content,
                self.answer.short_answer,
                self.answer.recommended_action,
                self.answer.rationale,
                self.answer.limitations,
            )):
                raise ValueError("negative outcome answer must be empty")
            return self._validate_canonical_content()
        source_numbers = {source.number for source in self.grounding.sources}
        sections = (
            self.answer.short_answer,
            self.answer.recommended_action,
            self.answer.rationale,
            self.answer.limitations,
        )
        valid = all(
            section.strip()
            and (references := {
                int(number) for number in re.findall(r"\[(\d+)\]", section)
            })
            and references <= source_numbers
            for section in sections
        )
        if not valid:
            raise ValueError(
                "structured answer sections must be nonempty and citation-backed"
            )
        return self._validate_canonical_content()

    def _validate_canonical_content(self) -> Self:
        if self.answer.content != _canonical_content(self.answer):
            raise ValueError("answer content must equal canonical structured sections")
        return self


class CoverageV2(WireModel):
    status: Literal["complete", "partial", "unavailable"]
    generation_id: str | None
    searched_documents: int = Field(ge=0)
    total_documents: int = Field(ge=0)
    incomplete_checks: int = Field(ge=0)

    @model_validator(mode="after")
    def facts_match_status(self) -> Self:
        if (
            self.generation_id is not None
            and GENERATION_ID_RE.fullmatch(self.generation_id) is None
        ):
            raise ValueError(
                "generation_id must be 64 lowercase hexadecimal characters"
            )
        if self.searched_documents > self.total_documents:
            raise ValueError("coverage searched_documents exceeds total_documents")
        if self.status == "complete" and (
            self.generation_id is None
            or self.total_documents == 0
            or self.searched_documents != self.total_documents
            or self.incomplete_checks != 0
        ):
            raise ValueError("complete coverage facts are contradictory")
        if self.status == "partial" and (
            self.generation_id is None
            or self.total_documents == 0
            or self.searched_documents == 0
            or (
                self.searched_documents == self.total_documents
                and self.incomplete_checks == 0
            )
        ):
            raise ValueError("partial coverage facts are contradictory")
        if self.status == "unavailable" and self.searched_documents != 0:
            raise ValueError("unavailable coverage facts are contradictory")
        return self


class ErrorV2(WireModel):
    kind: Literal["retrieval_unavailable", "synthesis_unavailable"]


class ChatErrorV2(WireModel):
    contract_version: Literal[2] = 2
    run_id: str
    error: ErrorV2
    coverage: CoverageV2

    @model_validator(mode="after")
    def error_kind_matches_coverage(self) -> Self:
        valid = (
            self.error.kind == "retrieval_unavailable"
            and self.coverage.status == "unavailable"
        ) or (
            self.error.kind == "synthesis_unavailable"
            and self.coverage.status in ("complete", "partial")
        )
        if not valid:
            raise ValueError("error kind contradicts coverage")
        return self


def _visual(visual) -> ExactVisualV2 | UnavailableVisualV2:
    if visual.status != "exact":
        return UnavailableVisualV2(
            reason=visual.reason or "location_unavailable"
        )
    grouped: dict[int, list[RegionV2]] = defaultdict(list)
    for region in visual.regions:
        grouped[region.page].append(RegionV2(
            x=region.x,
            y=region.y,
            width=region.width,
            height=region.height,
        ))
    return ExactVisualV2(pages=tuple(
        PageRegionsV2(page=page, regions=tuple(regions))
        for page, regions in sorted(grouped.items())
    ))


def _grounding_status(outcome: AnswerOutcome) -> str:
    if outcome.kind is AnswerKind.INSUFFICIENT_EVIDENCE:
        return "insufficient_evidence"
    if outcome.kind is AnswerKind.SEARCH_INCOMPLETE:
        return "search_incomplete"
    if outcome.coverage.status is LibraryStatus.PARTIAL:
        return "partially_grounded"
    return "grounded"


def _canonical_content(answer: AnswerV2) -> str:
    sections = (
        ("SHORT_ANSWER", answer.short_answer),
        ("RECOMMENDED_ACTION", answer.recommended_action),
        ("RATIONALE", answer.rationale),
        ("LIMITATIONS", answer.limitations),
    )
    if not any(value for _, value in sections):
        return ""
    return "\n".join(f"{label}: {value}" for label, value in sections)


def encode_outcome(
    outcome: AnswerOutcome, *, run_id: str
) -> ChatSuccessV2 | ChatErrorV2:
    """Encode one truthful Question answering outcome."""
    if outcome.kind in (
        AnswerKind.RETRIEVAL_UNAVAILABLE,
        AnswerKind.SYNTHESIS_UNAVAILABLE,
    ):
        coverage = outcome.coverage
        return ChatErrorV2(
            run_id=run_id,
            error=ErrorV2(kind=outcome.kind.value),
            coverage=CoverageV2(
                status=coverage.status.value,
                generation_id=coverage.generation_id,
                searched_documents=coverage.searched_documents,
                total_documents=coverage.total_documents,
                incomplete_checks=coverage.incomplete_checks,
            ),
        )
    answer = outcome.answer
    coverage = outcome.coverage
    answer_wire = AnswerV2(
        content="",
        short_answer=answer.short_answer if answer else "",
        recommended_action=answer.recommended_action if answer else "",
        rationale=answer.rationale if answer else "",
        limitations=answer.limitations if answer else "",
    )
    answer_wire = answer_wire.model_copy(
        update={"content": _canonical_content(answer_wire)}
    )
    return ChatSuccessV2(
        run_id=run_id,
        query=outcome.search.query,
        answer=answer_wire,
        grounding=GroundingV2(
            status=_grounding_status(outcome),
            generation_id=coverage.generation_id,
            searched_documents=coverage.searched_documents,
            total_documents=coverage.total_documents,
            incomplete_checks=coverage.incomplete_checks,
            sources=tuple(SourceV2(
                id=citation.id,
                number=citation.number,
                document=citation.document,
                node_id=citation.node_id,
                title=citation.title,
                breadcrumb=citation.breadcrumb,
                excerpt=citation.excerpt,
                quote=citation.quote,
                source_href=citation.source_href,
                visual=_visual(citation.visual),
            ) for citation in outcome.citations),
        ),
    )
