"""Authoritative producer tests for chat contract version 3."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from api.chat_wire import ChatResponseV3, RegionV3, encode_outcome
from api.question_answering import AnswerOutcome, EvidenceCitation, GroundedAnswer
from api.retrieval import (
    DocumentSearch,
    DocumentStatus,
    LibrarySearchResult,
    LibraryStatus,
    SearchDiagnostic,
    VerifiedEvidence,
)
from pageindex import PageNode
from pageindex.pins import NormalizedRegion, VisualLocation

GENERATION_ID = "a" * 64


def test_v3_region_cannot_extend_beyond_its_page():
    with pytest.raises(ValueError, match="stay within"):
        RegionV3(x=0.8, y=0.1, width=0.3, height=0.1)


def answered_outcome(*, exact: bool = False) -> AnswerOutcome:
    evidence = VerifiedEvidence(
        document="hygiene",
        node=PageNode(
            node_id="ppe",
            title="Schutzkleidung",
            heading_level=1,
            line_idx=0,
            summary="",
            content="Handschuhe tragen.",
        ),
        breadcrumb="Hygiene › Schutzkleidung",
        reason="nennt die Maßnahme",
        quote="Handschuhe tragen.",
    )
    document = DocumentSearch(
        "hygiene", DocumentStatus.SEARCHED, (), {}, (evidence,), ()
    )
    search = LibrarySearchResult(
        "Welche Maßnahmen bei MRSA?",
        GENERATION_ID,
        LibraryStatus.COMPLETE,
        (document,),
        (evidence,),
        (),
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
        source_href=f"/api/library/{GENERATION_ID}/documents/hygiene/pdf",
        visual=(
            VisualLocation("exact", (
                NormalizedRegion(2, 0.1, 0.2, 0.3, 0.04),
                NormalizedRegion(3, 0.2, 0.1, 0.4, 0.05),
            ))
            if exact
            else VisualLocation("unavailable", reason="missing_provenance_pin")
        ),
    )
    answer = GroundedAnswer(
        content="legacy canonical content is internal",
        short_answer="Handschuhe tragen [1].",
        recommended_action="Handschuhe tragen [1].",
        rationale="Die Dokumentstelle nennt Handschuhe [1].",
        limitations="Nur für diesen Kontext [1].",
    )
    return AnswerOutcome.answered(search, answer, (citation,))


def partial_search(answered: AnswerOutcome) -> LibrarySearchResult:
    diagnostic = SearchDiagnostic(
        "isolation", "document_unavailable", "technical state"
    )
    unavailable = DocumentSearch(
        "isolation", DocumentStatus.UNAVAILABLE, (), {}, (), (diagnostic,)
    )
    return replace(
        answered.search,
        status=LibraryStatus.PARTIAL,
        documents=(*answered.search.documents, unavailable),
        diagnostics=(diagnostic,),
    )


def test_an_answered_outcome_encodes_the_v3_practitioner_wire():
    wire = encode_outcome(answered_outcome(), run_id="run-test")

    assert wire.contract_version == 3
    assert wire.outcome.kind == "answered"
    assert wire.outcome.coverage.status == "complete"
    assert wire.outcome.citations[0].number == 1
    assert wire.outcome.citations[0].document_id == "hygiene"
    assert wire.outcome.citations[0].selection_reason == "nennt die Maßnahme"
    assert wire.outcome.citations[0].visual.pdf_href == (
        f"/api/library/{GENERATION_ID}/documents/hygiene/pdf"
    )
    assert wire.outcome.citations[0].visual.status == "unavailable"
    assert "content" not in wire.outcome.answer.model_dump()


def test_exact_visual_citation_binds_its_pdf_and_every_page():
    wire = encode_outcome(answered_outcome(exact=True), run_id="run-test")
    visual = wire.outcome.citations[0].visual

    assert visual.status == "exact"
    assert visual.pdf_href.endswith("/documents/hygiene/pdf")
    assert [page.page for page in visual.pages] == [2, 3]


def test_synthesis_unavailable_preserves_partial_coverage_and_citations():
    answered = answered_outcome()
    outcome = AnswerOutcome.synthesis_unavailable(
        partial_search(answered), answered.citations
    )

    wire = encode_outcome(outcome, run_id="run-test")

    assert wire.outcome.kind == "synthesis_unavailable"
    assert wire.outcome.coverage.status == "partial"
    assert wire.outcome.coverage.searched_documents == 1
    assert wire.outcome.coverage.total_documents == 2
    assert wire.outcome.citations[0].quote == "Handschuhe tragen."


def test_retrieval_unavailable_before_a_snapshot_has_null_generation():
    search = LibrarySearchResult(
        "Welche Maßnahmen bei MRSA?",
        None,
        LibraryStatus.UNAVAILABLE,
        (),
        (),
        (),
    )

    wire = encode_outcome(
        AnswerOutcome.retrieval_unavailable(search), run_id="run-test"
    )

    assert wire.outcome.kind == "retrieval_unavailable"
    assert wire.outcome.coverage.generation_id is None


def test_v3_wire_rejects_unknown_fields():
    payload = encode_outcome(
        answered_outcome(), run_id="run-test"
    ).model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs"):
        ChatResponseV3.model_validate(payload)


def test_exact_visual_requires_an_immutable_pdf_link():
    answered = answered_outcome(exact=True)
    citation = replace(answered.citations[0], source_href=None)
    outcome = AnswerOutcome.answered(
        answered.search, answered.answer, (citation,)
    )

    with pytest.raises(ValueError, match="immutable PDF"):
        encode_outcome(outcome, run_id="run-test")
