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
from .retrieval import LibrarySearchResult, LibraryStatus


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
    has_source_pdf: bool
    visual: VisualLocation


@dataclass(frozen=True)
class AnswerOutcome:
    kind: AnswerKind
    search: LibrarySearchResult
    answer: GroundedAnswer | None = None
    citations: tuple[EvidenceCitation, ...] = ()


class LibrarySearcher(Protocol):
    def search(self, query: str, run: Any) -> LibrarySearchResult: ...


class AnswerSynthesizer(Protocol):
    def synthesise(self, question: Question, evidence: tuple) -> GroundedAnswer: ...


class EvidenceCitationFactory:
    """Turn verified evidence into truthful textual and visual citations."""

    def __init__(
        self,
        page_size: Callable[[str, int], tuple[float, float] | None],
        source_documents: Callable[[], set[str]],
    ) -> None:
        self._page_size = page_size
        self._source_documents = source_documents

    def create(self, evidence: tuple) -> tuple[EvidenceCitation, ...]:
        source_documents = self._source_documents()
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
                    self._page_size,
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
                has_source_pdf=item.document in source_documents,
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
        citation_factory: EvidenceCitationFactory | None = None,
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
            return AnswerOutcome(AnswerKind.INSUFFICIENT_EVIDENCE, search)
        if search.status is LibraryStatus.PARTIAL and not search.evidence:
            return AnswerOutcome(AnswerKind.SEARCH_INCOMPLETE, search)
        if search.status is LibraryStatus.UNAVAILABLE:
            return AnswerOutcome(AnswerKind.RETRIEVAL_UNAVAILABLE, search)
        citations = (
            self._citation_factory.create(search.evidence)
            if self._citation_factory is not None
            else ()
        )
        try:
            answer = self._synthesizer.synthesise(question, search.evidence)
        except AnswerSynthesisUnavailable:
            return AnswerOutcome(
                AnswerKind.SYNTHESIS_UNAVAILABLE, search, citations=citations
            )
        cited = {int(value) for value in re.findall(r"\[(\d+)\]", answer.content)}
        if not cited or any(
            number < 1 or number > len(search.evidence) for number in cited
        ):
            return AnswerOutcome(
                AnswerKind.SYNTHESIS_UNAVAILABLE, search, citations=citations
            )
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
            return AnswerOutcome(
                AnswerKind.SYNTHESIS_UNAVAILABLE, search, citations=citations
            )
        return AnswerOutcome(AnswerKind.ANSWERED, search, answer, citations)
