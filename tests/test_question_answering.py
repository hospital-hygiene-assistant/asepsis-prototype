"""Question answering truth through its public interface."""

import pytest

from api.question_answering import (
    AnswerKind,
    AnswerOutcome,
    AnswerSynthesisUnavailable,
    EvidenceCitation,
    EvidenceCitationFactory,
    GroundedAnswer,
    PromptAnswerSynthesizer,
    Question,
    QuestionAnswering,
)
from api.retrieval import (
    DocumentSearch,
    DocumentStatus,
    LibrarySearchResult,
    LibraryStatus,
    SearchCoverage,
    SearchDiagnostic,
    VerifiedEvidence,
)
from pageindex import PageNode
from pageindex.pins import PixelBox, ProvenancePin, SourceSpan
from pageindex.pins import VisualLocation

GENERATION_ID = "a" * 64


class StubSearch:
    def __init__(self, result: LibrarySearchResult):
        self.result = result

    def search(self, query, run):
        assert query == "Welche Maßnahmen bei MRSA?"
        return self.result


class SynthesisMustNotRun:
    def synthesise(self, question, evidence):
        raise AssertionError("synthesis must not run without verified evidence")


class RecordingSynthesis:
    def __init__(self):
        self.calls = []

    def synthesise(self, question, evidence):
        self.calls.append((question, evidence))
        return GroundedAnswer(
            "Einzelzimmer gemäß Quelle [1].",
            short_answer="Einzelzimmer [1].",
            recommended_action="Einzelzimmer nutzen [1].",
            rationale="Die Quelle nennt die Maßnahme [1].",
            limitations="Nur für den beschriebenen Fall [1].",
        )


class UnavailableSynthesis:
    def synthesise(self, question, evidence):
        raise AnswerSynthesisUnavailable("Ollama synthesis model unavailable")


class InvalidCitationSynthesis:
    def synthesise(self, question, evidence):
        return GroundedAnswer("Diese Behauptung verweist auf eine fremde Quelle [2].")


class MissingCitationSynthesis:
    def synthesise(self, question, evidence):
        return GroundedAnswer("Diese Behauptung hat keinen Quellenverweis.")


def citation_factory() -> EvidenceCitationFactory:
    return EvidenceCitationFactory(
        page_size=lambda _document, _page: None,
        source_href=lambda _generation, _document: None,
    )


def test_question_answering_requires_a_citation_factory():
    with pytest.raises(TypeError, match="citation_factory"):
        QuestionAnswering(StubSearch(None), SynthesisMustNotRun())


def searched_document(evidence=()) -> DocumentSearch:
    return DocumentSearch(
        document="hygiene",
        status=DocumentStatus.SEARCHED,
        tree=(),
        node_meta={},
        evidence=tuple(evidence),
        diagnostics=(),
    )


def test_a_complete_search_without_evidence_is_an_honest_negative():
    document = DocumentSearch(
        document="hygiene",
        status=DocumentStatus.SEARCHED,
        tree=(),
        node_meta={},
        evidence=(),
        diagnostics=(),
    )
    search = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id=GENERATION_ID,
        status=LibraryStatus.COMPLETE,
        documents=(document,),
        evidence=(),
        diagnostics=(),
    )
    answering = QuestionAnswering(
        StubSearch(search), SynthesisMustNotRun(), citation_factory()
    )

    outcome = answering.answer(Question("Welche Maßnahmen bei MRSA?"))

    assert outcome.kind is AnswerKind.INSUFFICIENT_EVIDENCE
    assert outcome.answer is None
    assert outcome.search is search
    assert outcome.coverage == SearchCoverage(
        status=LibraryStatus.COMPLETE,
        generation_id=GENERATION_ID,
        searched_documents=1,
        total_documents=1,
        incomplete_checks=0,
    )


def test_an_outcome_rejects_impossible_complete_coverage():
    search = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id=GENERATION_ID,
        status=LibraryStatus.COMPLETE,
        documents=(),
        evidence=(),
        diagnostics=(),
    )

    with pytest.raises(ValueError, match="non-empty expected library"):
        AnswerOutcome.insufficient_evidence(search)


def test_an_outcome_rejects_a_noncanonical_generation_identity():
    search = LibrarySearchResult(
        query="q",
        generation_id="generation-test",
        status=LibraryStatus.UNAVAILABLE,
        documents=(),
        evidence=(),
        diagnostics=(),
    )

    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        AnswerOutcome.retrieval_unavailable(search)


@pytest.mark.parametrize(
    ("status", "documents", "diagnostics"),
    [
        (LibraryStatus.PARTIAL, (), ()),
        (LibraryStatus.PARTIAL, (
            DocumentSearch(
                "hygiene", DocumentStatus.SEARCHED, (), {}, (), ()
            ),
        ), ()),
        (LibraryStatus.PARTIAL, (
            DocumentSearch(
                "hygiene", DocumentStatus.UNAVAILABLE, (), {}, (), ()
            ),
        ), (SearchDiagnostic("hygiene", "unavailable", "technical"),)),
        (LibraryStatus.UNAVAILABLE, (
            DocumentSearch(
                "hygiene", DocumentStatus.SEARCHED, (), {}, (), ()
            ),
        ), ()),
    ],
)
def test_an_outcome_rejects_contradictory_incomplete_coverage(
    status, documents, diagnostics
):
    search = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id=GENERATION_ID,
        status=status,
        documents=documents,
        evidence=(),
        diagnostics=diagnostics,
    )

    with pytest.raises(ValueError, match="coverage"):
        AnswerOutcome.synthesis_unavailable(search, ())


def test_outcome_kinds_reject_contradictory_search_facts():
    searched = DocumentSearch(
        "hygiene", DocumentStatus.SEARCHED, (), {}, (), ()
    )
    unavailable = DocumentSearch(
        "isolation", DocumentStatus.UNAVAILABLE, (), {}, (), ()
    )
    diagnostic = SearchDiagnostic(
        "isolation", "document_unavailable", "technical"
    )
    complete = LibrarySearchResult(
        "q", GENERATION_ID, LibraryStatus.COMPLETE,
        (searched,), (), (),
    )
    partial = LibrarySearchResult(
        "q", GENERATION_ID, LibraryStatus.PARTIAL,
        (searched, unavailable), (), (diagnostic,),
    )
    unavailable_search = LibrarySearchResult(
        "q", None, LibraryStatus.UNAVAILABLE, (), (), (),
    )

    contradictions = (
        (AnswerOutcome.insufficient_evidence, (partial,)),
        (AnswerOutcome.search_incomplete, (complete,)),
        (AnswerOutcome.answered, (
            complete, GroundedAnswer("Behauptung [1]."), ()
        )),
        (AnswerOutcome.retrieval_unavailable, (complete,)),
        (AnswerOutcome.synthesis_unavailable, (unavailable_search, ())),
    )

    for factory, arguments in contradictions:
        with pytest.raises(ValueError, match="outcome"):
            factory(*arguments)


def test_an_outcome_rejects_a_citation_bound_to_different_evidence():
    evidence = VerifiedEvidence(
        "hygiene",
        PageNode(
            node_id="ppe",
            title="PPE",
            heading_level=1,
            line_idx=0,
            summary="",
            content="Wear gloves.",
        ),
        "Hygiene › PPE",
        "exact",
        "Wear gloves.",
    )
    search = LibrarySearchResult(
        "q",
        GENERATION_ID,
        LibraryStatus.COMPLETE,
        (searched_document((evidence,)),),
        (evidence,),
        (),
    )
    citation = EvidenceCitation(
        "s1",
        1,
        "different-document",
        "ppe",
        "PPE",
        "Hygiene › PPE",
        "Wear gloves.",
        "Wear gloves.",
        "exact",
        None,
        VisualLocation("unavailable", reason="missing_provenance_pin"),
    )

    with pytest.raises(ValueError, match="citation does not match verified evidence"):
        AnswerOutcome.answered(
            search,
            GroundedAnswer("Answer [1]."),
            (citation,),
        )


def test_an_incomplete_search_without_evidence_is_not_a_document_finding():
    diagnostic = SearchDiagnostic(
        "isolation", "document_unavailable", "technical"
    )
    search = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id=GENERATION_ID,
        status=LibraryStatus.PARTIAL,
        documents=(
            searched_document(),
            DocumentSearch(
                "isolation",
                DocumentStatus.UNAVAILABLE,
                (),
                {},
                (),
                (diagnostic,),
            ),
        ),
        evidence=(),
        diagnostics=(diagnostic,),
    )
    answering = QuestionAnswering(
        StubSearch(search), SynthesisMustNotRun(), citation_factory()
    )

    outcome = answering.answer(Question("Welche Maßnahmen bei MRSA?"))

    assert outcome.kind is AnswerKind.SEARCH_INCOMPLETE
    assert outcome.answer is None


def test_verified_evidence_is_the_only_input_to_answer_synthesis():
    node = PageNode(
        node_id="isolation",
        title="Isolation",
        heading_level=2,
        line_idx=1,
        summary="s",
        content="MRSA erfordert ein Einzelzimmer.",
    )
    evidence = VerifiedEvidence(
        document="hygiene",
        node=node,
        breadcrumb="Guideline › Isolation",
        reason="nennt die Maßnahme",
        quote="erfordert ein Einzelzimmer",
    )
    search = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id=GENERATION_ID,
        status=LibraryStatus.COMPLETE,
        documents=(searched_document((evidence,)),),
        evidence=(evidence,),
        diagnostics=(),
    )
    synthesis = RecordingSynthesis()
    answering = QuestionAnswering(StubSearch(search), synthesis, citation_factory())

    outcome = answering.answer(Question("Welche Maßnahmen bei MRSA?"))

    assert outcome.kind is AnswerKind.ANSWERED
    assert outcome.answer == GroundedAnswer(
        "Einzelzimmer gemäß Quelle [1].",
        short_answer="Einzelzimmer [1].",
        recommended_action="Einzelzimmer nutzen [1].",
        rationale="Die Quelle nennt die Maßnahme [1].",
        limitations="Nur für den beschriebenen Fall [1].",
    )
    assert synthesis.calls == [
        (Question("Welche Maßnahmen bei MRSA?"), (evidence,))
    ]


def test_a_synthesis_outage_is_a_visible_technical_outcome():
    node = PageNode(
        node_id="isolation",
        title="Isolation",
        heading_level=2,
        line_idx=1,
        summary="s",
        content="MRSA erfordert ein Einzelzimmer.",
    )
    evidence = VerifiedEvidence(
        document="hygiene",
        node=node,
        breadcrumb="Guideline › Isolation",
        reason="nennt die Maßnahme",
        quote="erfordert ein Einzelzimmer",
    )
    search = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id=GENERATION_ID,
        status=LibraryStatus.COMPLETE,
        documents=(searched_document((evidence,)),),
        evidence=(evidence,),
        diagnostics=(),
    )

    outcome = QuestionAnswering(
        StubSearch(search), UnavailableSynthesis(), citation_factory()
    ).answer(
        Question("Welche Maßnahmen bei MRSA?")
    )

    assert outcome.kind is AnswerKind.SYNTHESIS_UNAVAILABLE
    assert outcome.answer is None


def test_prompt_synthesis_numbers_verified_evidence_and_parses_answer_sections():
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        return (
            "SHORT_ANSWER: Einzelzimmer [1]\n"
            "RECOMMENDED_ACTION: Isolieren [1]\n"
            "RATIONALE: Die Leitlinie nennt dies [1]\n"
            "LIMITATIONS: Keine"
        )

    node = PageNode(
        node_id="isolation",
        title="Isolation",
        heading_level=2,
        line_idx=1,
        summary="s",
        content="MRSA erfordert ein Einzelzimmer.",
    )
    evidence = VerifiedEvidence(
        document="hygiene",
        node=node,
        breadcrumb="Isolation",
        reason="nennt die Maßnahme",
        quote="erfordert ein Einzelzimmer",
    )

    answer = PromptAnswerSynthesizer(complete).synthesise(
        Question("Welche Maßnahmen?", context="Erreger: MRSA"), (evidence,)
    )

    assert "[1] hygiene › Isolation" in prompts[0]
    assert "MRSA erfordert ein Einzelzimmer." in prompts[0]
    assert answer.short_answer == "Einzelzimmer [1]"
    assert answer.recommended_action == "Isolieren [1]"


def test_an_answer_with_an_out_of_range_citation_is_unavailable():
    node = PageNode(
        node_id="isolation",
        title="Isolation",
        heading_level=2,
        line_idx=1,
        summary="s",
        content="MRSA erfordert ein Einzelzimmer.",
    )
    evidence = VerifiedEvidence(
        document="hygiene",
        node=node,
        breadcrumb="Isolation",
        reason="nennt die Maßnahme",
        quote="erfordert ein Einzelzimmer",
    )
    search = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id=GENERATION_ID,
        status=LibraryStatus.COMPLETE,
        documents=(searched_document((evidence,)),),
        evidence=(evidence,),
        diagnostics=(),
    )

    outcome = QuestionAnswering(
        StubSearch(search), InvalidCitationSynthesis(), citation_factory()
    ).answer(
        Question("Welche Maßnahmen bei MRSA?")
    )

    assert outcome.kind is AnswerKind.SYNTHESIS_UNAVAILABLE
    assert outcome.answer is None


def test_an_answer_without_any_inline_citation_is_unavailable():
    evidence = VerifiedEvidence(
        document="hygiene",
        node=PageNode(
            node_id="isolation",
            title="Isolation",
            heading_level=2,
            line_idx=1,
            summary="s",
            content="MRSA erfordert ein Einzelzimmer.",
        ),
        breadcrumb="Isolation",
        reason="nennt die Maßnahme",
        quote="erfordert ein Einzelzimmer",
    )
    search = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id=GENERATION_ID,
        status=LibraryStatus.COMPLETE,
        documents=(searched_document((evidence,)),),
        evidence=(evidence,),
        diagnostics=(),
    )

    outcome = QuestionAnswering(
        StubSearch(search), MissingCitationSynthesis(), citation_factory()
    ).answer(
        Question("Welche Maßnahmen bei MRSA?")
    )

    assert outcome.kind is AnswerKind.SYNTHESIS_UNAVAILABLE
    assert outcome.answer is None


def test_citation_factory_exposes_only_an_exact_quote_location():
    page_size_calls = []

    def page_size(generation, document, page, pin_scale):
        page_size_calls.append((generation, document, page, pin_scale))
        return (1000.0, 2000.0)

    content = "Handschuhe tragen. Schutzkittel anlegen."
    evidence = VerifiedEvidence(
        document="hygiene",
        node=PageNode(
            node_id="schutzkleidung",
            title="Schutzkleidung",
            heading_level=2,
            line_idx=1,
            summary="s",
            content=content,
            pin=ProvenancePin(
                2,
                "hygiene",
                "schutzkleidung",
                (SourceSpan(4, 0, len(content), PixelBox(100, 200, 600, 400)),),
                2.0,
            ),
        ),
        breadcrumb="Hygieneplan › Schutzkleidung",
        reason="nennt die Maßnahme",
        quote="Handschuhe tragen.",
    )
    factory = EvidenceCitationFactory(
        page_size=page_size,
        source_href=lambda generation, document: (
            f"/api/library/{generation}/documents/{document}/pdf"
        ),
    )

    citation = factory.create((evidence,), GENERATION_ID)[0]

    assert citation.quote == "Handschuhe tragen."
    assert citation.source_href == (
        f"/api/library/{GENERATION_ID}/documents/hygiene/pdf"
    )
    assert citation.visual.status == "exact"
    assert citation.visual.regions[0].page == 4
    assert citation.visual.regions[0].x == 0.1
    assert page_size_calls == [(GENERATION_ID, "hygiene", 4, 2.0)]


def test_citation_factory_never_guesses_when_provenance_is_missing():
    evidence = VerifiedEvidence(
        document="markdown-sample",
        node=PageNode(
            node_id="ppe",
            title="PPE",
            heading_level=2,
            line_idx=1,
            summary="s",
            content="Wear gloves.",
        ),
        breadcrumb="PPE",
        reason="states PPE",
        quote="Wear gloves.",
    )
    factory = EvidenceCitationFactory(
        page_size=lambda _document, _page: (1000.0, 2000.0),
        source_href=lambda _generation, _document: None,
    )

    citation = factory.create((evidence,), GENERATION_ID)[0]

    assert citation.visual.status == "unavailable"
    assert citation.visual.reason == "missing_provenance_pin"
    assert citation.visual.regions == ()


def test_citation_factory_rejects_cross_document_provenance():
    node = PageNode(
        node_id="ppe",
        title="PPE",
        heading_level=2,
        line_idx=1,
        summary="s",
        content="Wear gloves.",
        pin=ProvenancePin(
            2,
            "different-document",
            "ppe",
            (SourceSpan(1, 0, 12, PixelBox(10, 20, 100, 80)),),
            2.0,
        ),
    )
    evidence = VerifiedEvidence(
        "claimed-document", node, "PPE", "relevant", "Wear gloves."
    )
    factory = EvidenceCitationFactory(
        page_size=lambda _document, _page: (1000.0, 2000.0),
        source_href=lambda _generation, _document: None,
    )

    citation = factory.create((evidence,), GENERATION_ID)[0]

    assert citation.visual.status == "unavailable"
    assert citation.visual.reason == "provenance_document_mismatch"


def test_synthesis_rejects_uncited_prose_outside_the_structured_sections():
    def complete(_prompt):
        return (
            "Isolate for 14 days.\n"
            "SHORT_ANSWER: Einzelzimmer [1]\n"
            "RECOMMENDED_ACTION: Isolieren [1]\n"
            "RATIONALE: Laut Quelle [1]\n"
            "LIMITATIONS: Nur diese Quelle [1]"
        )

    with pytest.raises(AnswerSynthesisUnavailable):
        PromptAnswerSynthesizer(complete).synthesise(
            Question("Welche Maßnahmen?"),
            (VerifiedEvidence(
                "hygiene",
                PageNode(
                    node_id="ppe", title="PPE", heading_level=2,
                    line_idx=1, summary="s", content="Einzelzimmer.",
                ),
                "PPE", "relevant", "Einzelzimmer.",
            ),),
        )
