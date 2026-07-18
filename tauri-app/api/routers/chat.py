"""Practitioner question answering through the one grounded workflow."""

from typing import Optional

import pageindex as _pi
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..answering_runtime import answer_question
from ..chat_wire import encode_outcome
from ..prompts import CHAT_SYNTHESIS_PROMPT
from ..question_answering import Question
from ..runs import DuplicateRunId, RunQueueFull, RunRegistryFull, registry

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

    def answer(run):
        run.begin_retrieval()
        outcome = answer_question(Question(query, req.context), run)
        return encode_outcome(outcome, run_id=run.id).model_dump(mode="json")

    try:
        accepted = registry.submit("chat", answer, req.run_id)
    except DuplicateRunId:
        return JSONResponse({"error": "run id already exists"}, status_code=409)
    except (RunQueueFull, RunRegistryFull):
        return JSONResponse({"error": "question queue is full"}, status_code=429)
    return JSONResponse(
        {"run_id": accepted["run_id"], "state": "queued"},
        status_code=202,
    )
