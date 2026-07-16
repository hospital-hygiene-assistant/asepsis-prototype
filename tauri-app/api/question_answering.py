"""Turn one truthful library search into a source-grounded answer.

The public interface owns the distinction between an honest negative, an
incomplete search, and a technical failure.  HTTP and CLI adapters may render
those outcomes differently, but cannot reinterpret them.
"""

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any, Callable, Protocol

from pageindex.pins import VisualLocation, locate_visual_citation

from .prompts import CHAT_SYNTHESIS_PROMPT, _context_block, _parse_answer_sections
from .retrieval import LibrarySearchResult, LibraryStatus, SearchCoverage


class AnswerKind(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SEARCH_INCOMPLETE = "search_incomplete"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"
    SYNTHESIS_UNAVAILABLE = "synthesis_unavailable"


class AnswerSynthesisUnavailable(RuntimeError):
    """The external synthesis model could not produce an answer."""


@dataclass(frozen=True)
class Question:
    text: str
    context: str | None = None


@dataclass(frozen=True)
class GroundedAnswer:
    content: str
    short_answer: str = ""
    recommended_action: str = ""
    rationale: str = ""
    limitations: str = ""


@dataclass(frozen=True)
class EvidenceCitation:
    id: str
    number: int
    document: str
    node_id: str
    title: str
    breadcrumb: str
    excerpt: str
    quote: str
    reason: str
    source_href: str | None
    visual: VisualLocation


@dataclass(frozen=True, init=False)
class AnswerOutcome:
    kind: AnswerKind
    search: LibrarySearchResult
    answer: GroundedAnswer | None = None
    citations: tuple[EvidenceCitation, ...] = ()

    @classmethod
    def _create(
        cls,
        kind: AnswerKind,
        search: LibrarySearchResult,
        answer: GroundedAnswer | None = None,
        citations: tuple[EvidenceCitation, ...] = (),
    ) -> "AnswerOutcome":
        outcome = object.__new__(cls)
        object.__setattr__(outcome, "kind", kind)
        object.__setattr__(outcome, "search", search)
        object.__setattr__(outcome, "answer", answer)
        object.__setattr__(outcome, "citations", citations)
        outcome.__post_init__()
        return outcome

    @classmethod
    def answered(
        cls,
        search: LibrarySearchResult,
        answer: GroundedAnswer,
        citations: tuple[EvidenceCitation, ...],
    ) -> "AnswerOutcome":
        return cls._create(AnswerKind.ANSWERED, search, answer, citations)

    @classmethod
    def insufficient_evidence(
        cls, search: LibrarySearchResult
    ) -> "AnswerOutcome":
        return cls._create(AnswerKind.INSUFFICIENT_EVIDENCE, search)

    @classmethod
    def search_incomplete(
        cls, search: LibrarySearchResult
    ) -> "AnswerOutcome":
        return cls._create(AnswerKind.SEARCH_INCOMPLETE, search)

    @classmethod
    def retrieval_unavailable(
        cls, search: LibrarySearchResult
    ) -> "AnswerOutcome":
        return cls._create(AnswerKind.RETRIEVAL_UNAVAILABLE, search)

    @classmethod
    def synthesis_unavailable(
        cls,
        search: LibrarySearchResult,
        citations: tuple[EvidenceCitation, ...],
    ) -> "AnswerOutcome":
        return cls._create(
            AnswerKind.SYNTHESIS_UNAVAILABLE,
            search,
            citations=citations,
        )

    def __post_init__(self) -> None:
        coverage = self.search.coverage
        has_evidence = bool(self.search.evidence)
        citations_match = (
            len(self.citations) == len(self.search.evidence)
            and all(
                citation.number == number
                and citation.document == evidence.document
                and citation.node_id == evidence.node.node_id
                and citation.title == evidence.node.title
                and citation.breadcrumb == evidence.breadcrumb
                and citation.excerpt == (evidence.node.content or "")
                and citation.quote == evidence.quote
                and citation.reason == evidence.reason
                for number, (evidence, citation) in enumerate(
                    zip(self.search.evidence, self.citations), start=1
                )
            )
        )
        if self.citations and not citations_match:
            raise ValueError("citation does not match verified evidence")
        has_all_citations = has_evidence and citations_match
        if self.kind is AnswerKind.INSUFFICIENT_EVIDENCE:
            valid = (
                coverage.status is LibraryStatus.COMPLETE
                and not has_evidence
                and self.answer is None
                and not self.citations
            )
        elif self.kind is AnswerKind.SEARCH_INCOMPLETE:
            valid = (
                coverage.status is LibraryStatus.PARTIAL
                and not has_evidence
                and self.answer is None
                and not self.citations
            )
        elif self.kind is AnswerKind.ANSWERED:
            valid = (
                coverage.status in (LibraryStatus.COMPLETE, LibraryStatus.PARTIAL)
                and has_evidence
                and self.answer is not None
                and has_all_citations
            )
        elif self.kind is AnswerKind.RETRIEVAL_UNAVAILABLE:
            valid = (
                coverage.status is LibraryStatus.UNAVAILABLE
                and not has_evidence
                and self.answer is None
                and not self.citations
            )
        else:
            valid = (
                coverage.status in (LibraryStatus.COMPLETE, LibraryStatus.PARTIAL)
                and has_evidence
                and self.answer is None
                and has_all_citations
            )
        if not valid:
            raise ValueError(
                f"{self.kind.value} outcome contradicts coverage, evidence, or answer"
            )

    @property
    def coverage(self) -> SearchCoverage:
        return self.search.coverage


class LibrarySearcher(Protocol):
    def search(self, query: str, run: Any) -> LibrarySearchResult: ...


class AnswerSynthesizer(Protocol):
    def synthesise(self, question: Question, evidence: tuple) -> GroundedAnswer: ...


class EvidenceCitationFactory:
    """Turn verified evidence into truthful textual and visual citations."""

    def __init__(
        self,
        page_size: Callable[[str, str, int, float], tuple[float, float] | None],
        source_href: Callable[[str, str], str | None],
    ) -> None:
        self._page_size = page_size
        self._source_href = source_href

    def create(
        self, evidence: tuple, generation_id: str
    ) -> tuple[EvidenceCitation, ...]:
        citations = []
        for number, item in enumerate(evidence, start=1):
            node = item.node
            if node.pin is None:
                visual = VisualLocation(
                    "unavailable", reason="missing_provenance_pin"
                )
            elif node.pin.document != item.document:
                visual = VisualLocation(
                    "unavailable", reason="provenance_document_mismatch"
                )
            else:
                visual = locate_visual_citation(
                    node.pin,
                    node.content or "",
                    item.quote,
                    lambda document, page: self._page_size(
                        generation_id, document, page, node.pin.scale
                    ),
                )
            citations.append(EvidenceCitation(
                id=f"s{number}",
                number=number,
                document=item.document,
                node_id=node.node_id,
                title=node.title,
                breadcrumb=item.breadcrumb,
                excerpt=node.content or "",
                quote=item.quote,
                reason=item.reason,
                source_href=self._source_href(generation_id, item.document),
                visual=visual,
            ))
        return tuple(citations)


class PromptAnswerSynthesizer:
    """Compose the one grounded synthesis prompt from verified evidence."""

    def __init__(self, complete) -> None:
        self._complete = complete

    def synthesise(self, question: Question, evidence: tuple) -> GroundedAnswer:
        passages = "\n\n".join(
            f"[{number}] {item.document} › {item.breadcrumb or item.node.title}\n"
            f"{item.node.content or '(no content)'}"
            for number, item in enumerate(evidence, start=1)
        )
        prompt = CHAT_SYNTHESIS_PROMPT.format(
            context_block=_context_block(question.context),
            query=question.text,
            passages=passages,
        )
        try:
            content = self._complete(prompt)
        except Exception as exc:
            raise AnswerSynthesisUnavailable(str(exc)) from exc
        content = re.sub(r"<thought>.*?(</thought>|$)", "", content, flags=re.S).strip()
        sections = _parse_answer_sections(content)
        if not re.match(
            r"^\s*(?:#+\s*)?\**\s*SHORT[\s_-]?ANSWER\s*\**\s*: ?",
            content,
            flags=re.IGNORECASE,
        ) or any(key not in sections for key in (
            "short_answer", "recommended_action", "rationale", "limitations"
        )):
            raise AnswerSynthesisUnavailable(
                "the synthesis response did not match the structured format"
            )
        content = "\n".join((
            f"SHORT_ANSWER: {sections['short_answer']}",
            f"RECOMMENDED_ACTION: {sections['recommended_action']}",
            f"RATIONALE: {sections['rationale']}",
            f"LIMITATIONS: {sections['limitations']}",
        ))
        return GroundedAnswer(
            content=content,
            short_answer=sections.get("short_answer", ""),
            recommended_action=sections.get("recommended_action", ""),
            rationale=sections.get("rationale", ""),
            limitations=sections.get("limitations", ""),
        )


class QuestionAnswering:
    def __init__(
        self,
        searcher: LibrarySearcher,
        synthesizer: AnswerSynthesizer,
        citation_factory: EvidenceCitationFactory,
    ) -> None:
        self._searcher = searcher
        self._synthesizer = synthesizer
        self._citation_factory = citation_factory

    def answer(self, question: Question, run: Any = None) -> AnswerOutcome:
        text = question.text.strip()
        if not text:
            raise ValueError("question must not be empty")
        search = self._searcher.search(text, run)
        if search.status is LibraryStatus.COMPLETE and not search.evidence:
            return AnswerOutcome.insufficient_evidence(search)
        if search.status is LibraryStatus.PARTIAL and not search.evidence:
            return AnswerOutcome.search_incomplete(search)
        if search.status is LibraryStatus.UNAVAILABLE:
            return AnswerOutcome.retrieval_unavailable(search)
        citations = self._citation_factory.create(
            search.evidence, search.generation_id
        )
        try:
            answer = self._synthesizer.synthesise(question, search.evidence)
        except AnswerSynthesisUnavailable:
            return AnswerOutcome.synthesis_unavailable(search, citations)
        cited = {int(value) for value in re.findall(r"\[(\d+)\]", answer.content)}
        if not cited or any(
            number < 1 or number > len(search.evidence) for number in cited
        ):
            return AnswerOutcome.synthesis_unavailable(search, citations)
        sections = (
            answer.short_answer,
            answer.recommended_action,
            answer.rationale,
            answer.limitations,
        )
        if any(
            not section
            or not re.findall(r"\[(\d+)\]", section)
            or any(
                int(number) < 1 or int(number) > len(search.evidence)
                for number in re.findall(r"\[(\d+)\]", section)
            )
            for section in sections
        ):
            return AnswerOutcome.synthesis_unavailable(search, citations)
        return AnswerOutcome.answered(search, answer, citations)
