"""Authoritative producer tests for the practitioner chat wire."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from api.chat_wire import (
    ChatErrorV2,
    ChatSuccessV2,
    CoverageV2,
    ErrorV2,
    RegionV2,
    encode_outcome,
)
from api.question_answering import (
    AnswerKind,
    AnswerOutcome,
    EvidenceCitation,
    GroundedAnswer,
)
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


def answered_outcome() -> AnswerOutcome:
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
        document="hygiene",
        status=DocumentStatus.SEARCHED,
        tree=(),
        node_meta={},
        evidence=(evidence,),
        diagnostics=(),
    )
    search = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id=GENERATION_ID,
        status=LibraryStatus.COMPLETE,
        documents=(document,),
        evidence=(evidence,),
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
        source_href=f"/api/library/{GENERATION_ID}/documents/hygiene/pdf",
        visual=VisualLocation("unavailable", reason="missing_provenance_pin"),
    )
    answer = GroundedAnswer(
        content="SHORT_ANSWER: Handschuhe tragen [1].",
        short_answer="Handschuhe tragen [1].",
        recommended_action="Handschuhe tragen [1].",
        rationale="Die Dokumentstelle nennt Handschuhe [1].",
        limitations="Nur für diesen Kontext [1].",
    )

    return AnswerOutcome(AnswerKind.ANSWERED, search, answer, (citation,))


def test_an_answered_outcome_encodes_the_v2_practitioner_wire():
    wire = encode_outcome(answered_outcome(), run_id="run-test")

    assert isinstance(wire, ChatSuccessV2)
    assert wire.contract_version == 2
    assert wire.grounding.status == "grounded"
    assert wire.grounding.sources[0].number == 1
    assert wire.grounding.sources[0].source_href == (
        f"/api/library/{GENERATION_ID}/documents/hygiene/pdf"
    )
    assert wire.grounding.sources[0].visual.status == "unavailable"


def test_the_v2_producer_rejects_noncontiguous_citation_numbers():
    payload = encode_outcome(
        answered_outcome(), run_id="run-test"
    ).model_dump(mode="json")
    payload["grounding"]["sources"][0]["number"] = 2

    with pytest.raises(ValidationError, match="contiguous"):
        ChatSuccessV2.model_validate(payload)


def test_synthesis_unavailable_wire_preserves_partial_coverage_not_grounding():
    answered = answered_outcome()
    unavailable = DocumentSearch(
        document="isolation",
        status=DocumentStatus.UNAVAILABLE,
        tree=(),
        node_meta={},
        evidence=(),
        diagnostics=(SearchDiagnostic(
            document="isolation",
            code="document_unavailable",
            message="technical state",
        ),),
    )
    search = replace(
        answered.search,
        status=LibraryStatus.PARTIAL,
        documents=(*answered.search.documents, unavailable),
        diagnostics=unavailable.diagnostics,
    )
    outcome = AnswerOutcome(
        AnswerKind.SYNTHESIS_UNAVAILABLE,
        search,
        citations=answered.citations,
    )

    wire = encode_outcome(outcome, run_id="run-test")

    assert isinstance(wire, ChatErrorV2)
    assert wire.error.kind == "synthesis_unavailable"
    assert wire.coverage.status == "partial"
    assert wire.coverage.searched_documents == 1
    assert wire.coverage.total_documents == 2
    assert "grounding" not in wire.model_dump(mode="json")
    assert "sources" not in wire.model_dump(mode="json")
    assert outcome.citations == answered.citations


def test_retrieval_unavailable_before_a_snapshot_encodes_null_generation():
    search = LibrarySearchResult(
        query="Welche Maßnahmen bei MRSA?",
        generation_id=None,
        status=LibraryStatus.UNAVAILABLE,
        documents=(),
        evidence=(),
        diagnostics=(),
    )

    wire = encode_outcome(
        AnswerOutcome(AnswerKind.RETRIEVAL_UNAVAILABLE, search),
        run_id="run-test",
    )

    assert isinstance(wire, ChatErrorV2)
    assert wire.coverage.generation_id is None


@pytest.mark.parametrize(
    ("kind", "status", "generation_id", "searched", "total", "incomplete"),
    [
        ("retrieval_unavailable", "complete", GENERATION_ID, 1, 1, 0),
        ("synthesis_unavailable", "unavailable", None, 0, 0, 0),
    ],
)
def test_error_wire_rejects_a_kind_that_contradicts_coverage(
    kind, status, generation_id, searched, total, incomplete
):
    with pytest.raises(ValidationError, match="error kind contradicts coverage"):
        ChatErrorV2(
            run_id="run-test",
            error=ErrorV2(kind=kind),
            coverage=CoverageV2(
                status=status,
                generation_id=generation_id,
                searched_documents=searched,
                total_documents=total,
                incomplete_checks=incomplete,
            ),
        )


def test_an_exact_visual_requires_a_generation_bound_source_href():
    outcome = answered_outcome()
    citation = replace(
        outcome.citations[0],
        source_href=None,
        visual=VisualLocation(
            "exact", (NormalizedRegion(1, 0.1, 0.2, 0.3, 0.04),)
        ),
    )

    with pytest.raises(ValidationError, match="exact visual requires source_href"):
        encode_outcome(
            replace(outcome, citations=(citation,)),
            run_id="run-test",
        )


def test_success_grounding_status_must_match_coverage_facts():
    payload = encode_outcome(
        answered_outcome(), run_id="run-test"
    ).model_dump(mode="json")
    payload["grounding"]["searched_documents"] = 0

    with pytest.raises(ValidationError, match="grounding status contradicts coverage"):
        ChatSuccessV2.model_validate(payload)


@pytest.mark.parametrize("short_answer", ["", "Handschuhe tragen."])
def test_grounded_answer_sections_must_be_nonempty_and_citation_backed(
    short_answer
):
    payload = encode_outcome(
        answered_outcome(), run_id="run-test"
    ).model_dump(mode="json")
    payload["answer"]["short_answer"] = short_answer

    with pytest.raises(ValidationError, match="structured answer sections"):
        ChatSuccessV2.model_validate(payload)


def test_a_clinical_negative_wire_cannot_contain_backend_prose():
    answered = answered_outcome()
    search = replace(
        answered.search,
        evidence=(),
        documents=(replace(answered.search.documents[0], evidence=()),),
    )
    outcome = AnswerOutcome(AnswerKind.INSUFFICIENT_EVIDENCE, search)
    payload = encode_outcome(outcome, run_id="run-test").model_dump(mode="json")
    payload["answer"]["short_answer"] = "Keine Evidenz gefunden."

    with pytest.raises(ValidationError, match="negative outcome answer must be empty"):
        ChatSuccessV2.model_validate(payload)


def test_a_wire_region_must_stay_inside_its_page():
    with pytest.raises(ValidationError, match="stay within its page"):
        RegionV2(x=0.8, y=0.9, width=0.3, height=0.2)


@pytest.mark.parametrize(
    "source_href",
    [
        "",
        f"https://example.test/api/library/{GENERATION_ID}/documents/hygiene/pdf",
        "/api/document/hygiene/pdf",
        "/api/library//documents/hygiene/pdf",
    ],
)
def test_source_href_must_be_the_canonical_generation_scoped_route(source_href):
    payload = encode_outcome(
        answered_outcome(), run_id="run-test"
    ).model_dump(mode="json")
    payload["grounding"]["sources"][0]["source_href"] = source_href

    with pytest.raises(ValidationError, match="canonical generation-scoped route"):
        ChatSuccessV2.model_validate(payload)


def test_noncanonical_generation_identity_is_rejected():
    with pytest.raises(ValidationError, match="64 lowercase hexadecimal"):
        CoverageV2(
            status="complete",
            generation_id="generation-test",
            searched_documents=1,
            total_documents=1,
            incomplete_checks=0,
        )


def test_source_href_must_name_the_grounding_generation():
    payload = encode_outcome(
        answered_outcome(), run_id="run-test"
    ).model_dump(mode="json")
    payload["grounding"]["sources"][0]["source_href"] = (
        f"/api/library/{'b' * 64}/documents/hygiene/pdf"
    )

    with pytest.raises(ValidationError, match="grounding generation"):
        ChatSuccessV2.model_validate(payload)


def test_answer_content_is_the_canonical_structured_sections():
    wire = encode_outcome(answered_outcome(), run_id="run-test")
    expected = (
        "SHORT_ANSWER: Handschuhe tragen [1].\n"
        "RECOMMENDED_ACTION: Handschuhe tragen [1].\n"
        "RATIONALE: Die Dokumentstelle nennt Handschuhe [1].\n"
        "LIMITATIONS: Nur für diesen Kontext [1]."
    )
    assert wire.answer.content == expected

    payload = wire.model_dump(mode="json")
    payload["answer"]["content"] = "Widersprüchlicher Freitext [1]."
    with pytest.raises(ValidationError, match="canonical structured sections"):
        ChatSuccessV2.model_validate(payload)
