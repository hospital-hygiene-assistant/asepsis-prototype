"""Practitioner question answering through the one grounded workflow."""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import pageindex as _pi
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..pdf import _rendered_page_size
from ..prompts import CHAT_SYNTHESIS_PROMPT
from ..question_answering import (
    AnswerKind,
    AnswerOutcome,
    EvidenceCitationFactory,
    PromptAnswerSynthesizer,
    Question,
    QuestionAnswering,
)
from ..retrieval import LibraryStatus, WholeLibraryRetrieval
from ..runs import registry
from ..sources import load_sources

router = APIRouter()


@router.get("/api/chat/config")
def chat_config():
    """Expose the exact models and prompts used by the answering module."""
    activity = _pi.get_activity()
    return JSONResponse({
        "retrieval_model": _pi.settings.model,
        "synthesis_model": _pi.settings.synthesis_model,
        "temperature": 0,
        "ollama_urls": _pi.OLLAMA_URLS,
        "ollama_instances": len(_pi.OLLAMA_URLS),
        "any_busy": any(value > 0 for value in activity.values()),
        "prompts": {
            "synthesis": CHAT_SYNTHESIS_PROMPT,
            "section_pruning": _pi.SECTION_CHECK_PROMPT,
            "leaf_evaluation": _pi.LEAF_EVAL_PROMPT,
            "why_not_explainer": _pi.EXPLAIN_PROMPT,
        },
    })


class ChatRequest(BaseModel):
    run_id: Optional[str] = None
    query: str
    context: Optional[str] = None


def _question_answering(run):
    """Snapshot model selection once and assemble the answering module."""
    synthesis_model = _pi.settings.synthesis_model
    synthesis_url = _pi.OLLAMA_URLS[0]
    synthesis_client = _pi.make_client(synthesis_url)

    def complete(prompt: str) -> str:
        run.set_phase("synthesis")
        response = synthesis_client.chat(
            model=synthesis_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        return response["message"]["content"]

    citations = EvidenceCitationFactory(
        page_size=_rendered_page_size,
        source_documents=lambda: {
            stem
            for stem, source in load_sources().items()
            if Path(str(source.get("pdf") or "")).is_file()
        },
    )
    return QuestionAnswering(
        WholeLibraryRetrieval(_pi),
        PromptAnswerSynthesizer(complete),
        citations,
    )


def _visual_wire(visual) -> dict:
    if visual.status != "exact":
        return {
            "status": "unavailable",
            "reason": visual.reason or "location_unavailable",
        }
    grouped = defaultdict(list)
    for region in visual.regions:
        grouped[region.page].append({
            "x": region.x,
            "y": region.y,
            "width": region.width,
            "height": region.height,
        })
    return {
        "status": "exact",
        "pages": [
            {"page": page, "regions": regions}
            for page, regions in sorted(grouped.items())
        ],
    }


def _grounding_status(outcome: AnswerOutcome) -> str:
    if outcome.kind is AnswerKind.INSUFFICIENT_EVIDENCE:
        return "insufficient_evidence"
    if outcome.kind is AnswerKind.SEARCH_INCOMPLETE:
        return "search_incomplete"
    if outcome.search.status is LibraryStatus.PARTIAL:
        return "partially_grounded"
    return "grounded"


def _answer_wire(outcome: AnswerOutcome, run_id: str) -> dict:
    answer = outcome.answer
    searched = sum(
        document.status.value != "unavailable"
        for document in outcome.search.documents
    )
    return {
        "contract_version": 2,
        "run_id": run_id,
        "query": outcome.search.query,
        "answer": {
            "content": answer.content if answer else "",
            "short_answer": answer.short_answer if answer else "",
            "recommended_action": answer.recommended_action if answer else "",
            "rationale": answer.rationale if answer else "",
            "limitations": answer.limitations if answer else "",
        },
        "grounding": {
            "status": _grounding_status(outcome),
            "generation_id": outcome.search.generation_id,
            "searched_documents": searched,
            "total_documents": len(outcome.search.documents),
            "incomplete_checks": len(outcome.search.diagnostics),
            "sources": [
                {
                    "id": citation.id,
                    "number": citation.number,
                    "document": citation.document,
                    "node_id": citation.node_id,
                    "title": citation.title,
                    "breadcrumb": citation.breadcrumb,
                    "excerpt": citation.excerpt,
                    "quote": citation.quote,
                    "has_source_pdf": citation.has_source_pdf,
                    "visual": _visual_wire(citation.visual),
                }
                for citation in outcome.citations
            ],
        },
    }


@router.post("/api/chat")
def chat(req: ChatRequest):
    query = (req.query or "").strip()
    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)

    run = registry.create(req.run_id)
    run.set_phase("retrieval")
    try:
        answering = _question_answering(run)
        outcome = answering.answer(Question(query, req.context), run)
        if outcome.kind in (
            AnswerKind.RETRIEVAL_UNAVAILABLE,
            AnswerKind.SYNTHESIS_UNAVAILABLE,
        ):
            run.set_phase("error")
            return JSONResponse(
                {
                    "contract_version": 2,
                    "error": {"kind": outcome.kind.value},
                    "run_id": run.id,
                },
                status_code=503,
            )
        return JSONResponse(_answer_wire(outcome, run.id))
    except Exception:
        kind = (
            "synthesis_unavailable"
            if run.phase == "synthesis"
            else "retrieval_unavailable"
        )
        run.set_phase("error")
        return JSONResponse(
            {
                "contract_version": 2,
                "error": {"kind": kind},
                "run_id": run.id,
            },
            status_code=503,
        )
    finally:
        if run.phase != "error":
            run.set_phase("idle")
