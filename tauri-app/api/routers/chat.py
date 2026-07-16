"""Practitioner question answering through the one grounded workflow."""

from typing import Optional

import pageindex as _pi
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..answering_runtime import build_question_answering
from ..chat_wire import ChatErrorV2, encode_outcome
from ..prompts import CHAT_SYNTHESIS_PROMPT
from ..question_answering import (
    AnswerKind,
    AnswerOutcome,
    Question,
)
from ..retrieval import (
    LibrarySearchResult,
    LibraryStatus,
    SearchDiagnostic,
)
from ..runs import registry

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


@router.post("/api/chat")
def chat(req: ChatRequest):
    query = (req.query or "").strip()
    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)

    run = registry.create(req.run_id)
    run.set_phase("retrieval")
    try:
        try:
            answering = build_question_answering(run)
        except Exception:
            diagnostic = SearchDiagnostic(
                document="__runtime__",
                code="question_answering_unavailable",
                message="question answering runtime failed",
            )
            outcome = AnswerOutcome(
                AnswerKind.RETRIEVAL_UNAVAILABLE,
                LibrarySearchResult(
                    query=query,
                    generation_id=None,
                    status=LibraryStatus.UNAVAILABLE,
                    documents=(),
                    evidence=(),
                    diagnostics=(diagnostic,),
                ),
            )
            wire = encode_outcome(outcome, run_id=run.id)
            run.set_phase("error")
            return JSONResponse(
                wire.model_dump(mode="json"),
                status_code=503,
            )
        try:
            outcome = answering.answer(Question(query, req.context), run)
            wire = encode_outcome(outcome, run_id=run.id)
        except Exception:
            run.set_phase("error")
            raise
        if isinstance(wire, ChatErrorV2):
            run.set_phase("error")
            return JSONResponse(wire.model_dump(mode="json"), status_code=503)
        return JSONResponse(wire.model_dump(mode="json"))
    finally:
        if run.phase != "error":
            run.set_phase("idle")
