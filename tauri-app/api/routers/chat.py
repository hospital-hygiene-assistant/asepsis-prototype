"""The practitioner chat: retrieval, synthesis, and its grounding verdict."""

import re
from typing import Optional

import pageindex as _pi
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from modules.registry import load as _load_module, defaults as _module_defaults

from ..pdf import _normalized_bbox
from ..prompts import CHAT_SYNTHESIS_PROMPT, _context_block, _parse_answer_sections
from ..retrieval import breadcrumbs_for, count_eval_errors, run_retrieval
from ..runs import registry
from ..sources import load_sources

router = APIRouter()


# ---------------------------------------------------------------------------
# Chat — retrieval + grounded answer synthesis for the chatbot tab
# ---------------------------------------------------------------------------




@router.get("/api/chat/config")
def chat_config():
    """Introspection for the chatbot tab: which model answers, over which
    engine pool, with exactly which prompts — nothing hidden."""
    activity = _pi.get_activity()
    defs = _module_defaults()
    return JSONResponse({
        "model": _pi.settings.model,
        "retrieval_model": _pi.settings.model,
        "synthesis_model": _pi.settings.synthesis_model,
        "temperature": 0,
        "ollama_urls": _pi.OLLAMA_URLS,
        "ollama_instances": len(_pi.OLLAMA_URLS),
        "any_busy": any(v > 0 for v in activity.values()),
        "pipeline": defs,
        "prompts": {
            "synthesis": CHAT_SYNTHESIS_PROMPT,
            "section_pruning": getattr(_pi, "SECTION_CHECK_PROMPT", ""),
            "leaf_evaluation": getattr(_pi, "LEAF_EVAL_PROMPT", ""),
            "why_not_explainer": getattr(_pi, "EXPLAIN_PROMPT", ""),
        },
    })


class ChatRequest(BaseModel):
    # The client's own id for this run, so it can poll /api/runs/{run_id} from
    # the moment it sends this rather than waiting to be told what to watch.
    run_id: Optional[str] = None
    query: str
    index_module: Optional[str] = None
    # Pre-rendered, human-readable background from a client-side structured
    # intake. Kept as free text so the server stays agnostic about which
    # questionnaire produced it.
    context: Optional[str] = None


@router.post("/api/chat")
def chat(req: ChatRequest):
    """The chatbot workflow: the same two-phase retrieval as /api/run, then a
    synthesis call that produces a structured, citation-anchored answer.

    The response carries both the chat payload (answer + sources) and the full
    run payload, so the retrieval tab can render the identical run state.
    """
    query = (req.query or "").strip()
    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)
    patient_context = _context_block(req.context)
    run = registry.create(req.run_id)

    defs = _module_defaults()
    index_mod_name = req.index_module or defs["index"]
    try:
        index_mod = _load_module("index", index_mod_name)
    except Exception as exc:
        return JSONResponse({"error": f"Could not load index module '{index_mod_name}': {exc}"}, status_code=400)

    run.set_phase("retrieval", "Reading the document trees…")
    try:
        results = run_retrieval(query, index_mod, run)
        eval_errors = count_eval_errors(results)

        # Nothing retrieved *and* nothing successfully evaluated means retrieval
        # never ran — the model was unreachable. Reporting that as "no passage
        # was judged relevant" states a clinical finding the system never made,
        # and a practitioner could reasonably read it as "no guideline covers
        # this". Fail loudly, before doing any more work on an empty run.
        if not any(doc_data.get("nodes") for doc_data in results.values()) and eval_errors:
            run.set_phase("error", "Retrieval could not run")
            return JSONResponse(
                {"error": "The retrieval model is unavailable, so the library "
                          "could not be searched. This is not a finding about "
                          "the documents."},
                status_code=503)

        # Flatten retrieved nodes into numbered sources (stable order: doc, tree order)
        crumbs = breadcrumbs_for(results)
        sources = []
        for doc_name, doc_data in results.items():
            for node in doc_data["nodes"]:
                pin = node.get("pin") or {}
                sources.append({
                    "id": f"s{len(sources) + 1}",
                    "n": len(sources) + 1,
                    "doc": doc_name,
                    "node_id": node["node_id"],
                    "title": node["title"],
                    "breadcrumb": crumbs.get(doc_name, {}).get(node["node_id"], ""),
                    "excerpt": node["content"],
                    "quote": node["quote"],
                    "reason": node["reason"],
                    "synthetic": node["synthetic"],
                    "pin": node.get("pin"),
                    "page": pin.get("page"),
                    "bbox_normalized": _normalized_bbox(doc_name, pin) if pin else None,
                    "image": pin.get("image"),
                    "has_source_pdf": doc_name in load_sources(),
                })

        answer_text = ""
        sections: dict = {}
        if sources:
            run.set_phase("synthesis", f"Composing an answer from {len(sources)} passages…")
            passages = "\n\n".join(
                f"[{s['n']}] {s['doc'].replace('_', ' ')} › {s['breadcrumb'] or s['title']}\n{s['excerpt'] or '(no content)'}"
                for s in sources
            )
            prompt = CHAT_SYNTHESIS_PROMPT.format(
                context_block=patient_context, query=query, passages=passages,
            )
            client = _pi.make_client(_pi.OLLAMA_URLS[0])
            response = client.chat(
                model=_pi.settings.synthesis_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            answer_text = response["message"]["content"]
            # Strip thought blocks
            answer_text = re.sub(r"<thought>.*?(</thought>|$)", "", answer_text, flags=re.S).strip()
            sections = _parse_answer_sections(answer_text)

        cited = set(int(n) for n in re.findall(r"\[(\d+)\]", answer_text))
        valid_cited = {n for n in cited if 1 <= n <= len(sources)}
        if not sources:
            status = "insufficient_evidence"
            summary = "No passage in the library was judged relevant to this question."
        elif valid_cited:
            status = "grounded"
            summary = (f"{len(valid_cited)} of {len(sources)} retrieved passages are "
                       f"cited inline; every source below was selected with a verbatim quote.")
        else:
            status = "partially_grounded"
            summary = (f"{len(sources)} passages were retrieved, but the answer text "
                       f"carries no inline citations — verify against the sources below.")

        if eval_errors:
            # Some of the library was never actually read. Say so rather than
            # letting the summary imply the whole corpus was considered.
            status = "partially_grounded"
            summary = (f"{summary} {eval_errors} passage(s) could not be checked "
                       f"because the model was unavailable, so the library was not "
                       f"fully searched.")

        return JSONResponse({
            "run_id": run.id,
            "query": query,
            "answer": {
                "content": answer_text,
                "short_answer": sections.get("short_answer", ""),
                "recommended_action": sections.get("recommended_action", ""),
                "rationale": sections.get("rationale", ""),
                "limitations": sections.get("limitations", ""),
            },
            "grounding": {"status": status, "summary": summary, "sources": sources},
            "run": {
                "query": query,
                "results": results,
                "test_result": None,
                "pipeline": {
                    "ingest": defs["ingest"],
                    "index":  index_mod_name,
                    "query":  defs["query"],
                },
            },
        })
    except Exception as exc:
        run.set_phase("error", str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        # An error phase is the answer to the request; leave it for the client
        # to read rather than resetting it out from under them.
        if run.phase != "error":
            run.set_phase("idle")
