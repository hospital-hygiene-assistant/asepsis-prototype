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
from api.retrieval import LibrarySearchResult, LibraryStatus, VerifiedEvidence
from pageindex import PageNode
from pageindex.pins import PixelBox, ProvenancePin, SourceSpan
from pageindex.pins import NormalizedRegion, VisualLocation
from api.routers.chat import _answer_wire


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


def test_a_complete_search_without_evidence_is_an_honest_negative():
    search = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id="generation-test",
        status=LibraryStatus.COMPLETE,
        documents=(),
        evidence=(),
        diagnostics=(),
    )
    answering = QuestionAnswering(StubSearch(search), SynthesisMustNotRun())

    outcome = answering.answer(Question("Welche Maßnahmen bei MRSA?"))

    assert outcome.kind is AnswerKind.INSUFFICIENT_EVIDENCE
    assert outcome.answer is None
    assert outcome.search is search


def test_an_incomplete_search_without_evidence_is_not_a_document_finding():
    search = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id="generation-test",
        status=LibraryStatus.PARTIAL,
        documents=(),
        evidence=(),
        diagnostics=(),
    )
    answering = QuestionAnswering(StubSearch(search), SynthesisMustNotRun())

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
        generation_id="generation-test",
        status=LibraryStatus.COMPLETE,
        documents=(),
        evidence=(evidence,),
        diagnostics=(),
    )
    synthesis = RecordingSynthesis()
    answering = QuestionAnswering(StubSearch(search), synthesis)

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
        generation_id="generation-test",
        status=LibraryStatus.COMPLETE,
        documents=(),
        evidence=(evidence,),
        diagnostics=(),
    )

    outcome = QuestionAnswering(StubSearch(search), UnavailableSynthesis()).answer(
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
        generation_id="generation-test",
        status=LibraryStatus.COMPLETE,
        documents=(),
        evidence=(evidence,),
        diagnostics=(),
    )

    outcome = QuestionAnswering(StubSearch(search), InvalidCitationSynthesis()).answer(
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
        generation_id="generation-test",
        status=LibraryStatus.COMPLETE,
        documents=(),
        evidence=(evidence,),
        diagnostics=(),
    )

    outcome = QuestionAnswering(StubSearch(search), MissingCitationSynthesis()).answer(
        Question("Welche Maßnahmen bei MRSA?")
    )

    assert outcome.kind is AnswerKind.SYNTHESIS_UNAVAILABLE
    assert outcome.answer is None


def test_citation_factory_exposes_only_an_exact_quote_location():
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
        page_size=lambda _document, _page: (1000.0, 2000.0),
        source_documents=lambda: {"hygiene"},
    )

    citation = factory.create((evidence,))[0]

    assert citation.quote == "Handschuhe tragen."
    assert citation.has_source_pdf is True
    assert citation.visual.status == "exact"
    assert citation.visual.regions[0].page == 4
    assert citation.visual.regions[0].x == 0.1


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
        source_documents=set,
    )

    citation = factory.create((evidence,))[0]

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
        source_documents=lambda: {"claimed-document", "different-document"},
    )

    citation = factory.create((evidence,))[0]

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


def test_v2_wire_exposes_facts_and_exact_locations_without_legacy_geometry():
    result = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id="generation-test",
        status=LibraryStatus.COMPLETE,
        documents=(),
        evidence=(),
        diagnostics=(),
    )
    citation = EvidenceCitation(
        id="s1",
        number=1,
        document="hygiene",
        node_id="ppe",
        title="Schutzkleidung",
        breadcrumb="Hygiene › Schutzkleidung",
        excerpt="Handschuhe tragen.",
        quote="Handschuhe tragen.",
        reason="nennt die Maßnahme",
        has_source_pdf=True,
        visual=VisualLocation(
            "exact", (NormalizedRegion(4, 0.1, 0.2, 0.5, 0.05),)
        ),
    )
    outcome = AnswerOutcome(
        AnswerKind.ANSWERED,
        result,
        GroundedAnswer("Handschuhe tragen [1]."),
        (citation,),
    )

    wire = _answer_wire(outcome, "run-test")

    assert wire["contract_version"] == 2
    assert "summary" not in wire["grounding"]
    source = wire["grounding"]["sources"][0]
    assert source["visual"] == {
        "status": "exact",
        "pages": [{
            "page": 4,
            "regions": [{"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.05}],
        }],
    }
    assert "page" not in source
    assert "bbox" not in source
    assert "bbox_normalized" not in source
