"""Deterministic examples emitted through the authoritative chat producer."""

import argparse
from pathlib import Path
import sys

from pageindex import PageNode
from pageindex.pins import NormalizedRegion, VisualLocation

from .chat_wire import encode_outcome
from .question_answering import (
    AnswerKind,
    AnswerOutcome,
    EvidenceCitation,
    GroundedAnswer,
)
from .retrieval import (
    DocumentSearch,
    DocumentStatus,
    LibrarySearchResult,
    LibraryStatus,
    SearchDiagnostic,
    VerifiedEvidence,
)

GENERATION_ID = "a" * 64
QUERY = "Welche Maßnahmen gelten?"


def _evidence() -> VerifiedEvidence:
    return VerifiedEvidence(
        document="hygiene",
        node=PageNode(
            node_id="ppe",
            title="Schutzkleidung",
            heading_level=1,
            line_idx=0,
            summary="",
            content="Verifizierte Dokumentstelle.",
        ),
        breadcrumb="Hygiene › Schutzkleidung",
        reason="verifizierte Auswahl",
        quote="Verifizierte Dokumentstelle.",
    )


def _citation(*, exact: bool = False) -> EvidenceCitation:
    return EvidenceCitation(
        id="s1",
        number=1,
        document="hygiene",
        node_id="ppe",
        title="Schutzkleidung",
        breadcrumb="Hygiene › Schutzkleidung",
        excerpt="Verifizierte Dokumentstelle.",
        quote="Verifizierte Dokumentstelle.",
        reason="verifizierte Auswahl",
        source_href=f"/api/library/{GENERATION_ID}/documents/hygiene/pdf",
        visual=(
            VisualLocation("exact", (
                NormalizedRegion(2, 0.1, 0.2, 0.5, 0.08),
                NormalizedRegion(3, 0.12, 0.15, 0.45, 0.07),
            ))
            if exact
            else VisualLocation("unavailable", reason="page_size_unavailable")
        ),
    )


def _answer() -> GroundedAnswer:
    return GroundedAnswer(
        content=(
            "SHORT_ANSWER: Belegte Aussage [1].\n"
            "RECOMMENDED_ACTION: Belegte Handlung [1].\n"
            "RATIONALE: Belegte Begründung [1].\n"
            "LIMITATIONS: Belegte Einschränkung [1]."
        ),
        short_answer="Belegte Aussage [1].",
        recommended_action="Belegte Handlung [1].",
        rationale="Belegte Begründung [1].",
        limitations="Belegte Einschränkung [1].",
    )


def _complete(evidence: tuple[VerifiedEvidence, ...]) -> LibrarySearchResult:
    document = DocumentSearch(
        "hygiene", DocumentStatus.SEARCHED, (), {}, evidence, ()
    )
    return LibrarySearchResult(
        QUERY,
        GENERATION_ID,
        LibraryStatus.COMPLETE,
        (document,),
        evidence,
        (),
    )


def _partial(evidence: tuple[VerifiedEvidence, ...]) -> LibrarySearchResult:
    diagnostic = SearchDiagnostic(
        "isolation", "document_unavailable", "technical state"
    )
    searched = DocumentSearch(
        "hygiene", DocumentStatus.SEARCHED, (), {}, evidence, ()
    )
    unavailable = DocumentSearch(
        "isolation",
        DocumentStatus.UNAVAILABLE,
        (),
        {},
        (),
        (diagnostic,),
    )
    return LibrarySearchResult(
        QUERY,
        GENERATION_ID,
        LibraryStatus.PARTIAL,
        (searched, unavailable),
        evidence,
        (diagnostic,),
    )


def fixture_payloads() -> dict[str, str]:
    evidence = (_evidence(),)
    citation = (_citation(),)
    exact_citation = (_citation(exact=True),)
    outcomes = {
        "answered_complete.json": AnswerOutcome(
            AnswerKind.ANSWERED, _complete(evidence), _answer(), exact_citation
        ),
        "answered_partial.json": AnswerOutcome(
            AnswerKind.ANSWERED, _partial(evidence), _answer(), citation
        ),
        "insufficient_evidence.json": AnswerOutcome(
            AnswerKind.INSUFFICIENT_EVIDENCE, _complete(())
        ),
        "search_incomplete.json": AnswerOutcome(
            AnswerKind.SEARCH_INCOMPLETE, _partial(())
        ),
        "retrieval_unavailable.json": AnswerOutcome(
            AnswerKind.RETRIEVAL_UNAVAILABLE,
            LibrarySearchResult(
                QUERY, None, LibraryStatus.UNAVAILABLE, (), (), ()
            ),
        ),
        "synthesis_unavailable_complete.json": AnswerOutcome(
            AnswerKind.SYNTHESIS_UNAVAILABLE,
            _complete(evidence),
            citations=citation,
        ),
        "synthesis_unavailable_partial.json": AnswerOutcome(
            AnswerKind.SYNTHESIS_UNAVAILABLE,
            _partial(evidence),
            citations=citation,
        ),
    }
    return {
        filename: encode_outcome(
            outcome, run_id=f"fixture-{filename.removesuffix('.json')}"
        ).model_dump_json(indent=2) + "\n"
        for filename, outcome in outcomes.items()
    }


def _write(targets: list[Path]) -> None:
    payloads = fixture_payloads()
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        for existing in target.iterdir():
            if existing.is_file() and existing.name not in payloads:
                existing.unlink()
        for filename, payload in payloads.items():
            (target / filename).write_text(payload, encoding="utf-8")


def _mismatches(targets: list[Path]) -> list[str]:
    payloads = fixture_payloads()
    mismatches: list[str] = []
    for target in targets:
        actual_names = {
            path.name for path in target.iterdir() if path.is_file()
        } if target.is_dir() else set()
        if actual_names != set(payloads):
            mismatches.append(f"{target}: fixture file set differs")
        for filename, payload in payloads.items():
            path = target / filename
            if not path.is_file() or path.read_text(encoding="utf-8") != payload:
                mismatches.append(f"{path}: content differs")
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write or verify authoritative v2 chat fixtures."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("targets", nargs="+", type=Path)
    args = parser.parse_args(argv)

    if args.write:
        _write(args.targets)
        return 0
    mismatches = _mismatches(args.targets)
    for mismatch in mismatches:
        print(mismatch, file=sys.stderr)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
