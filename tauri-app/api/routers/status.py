"""Engine state: pool, models, modules, and the retrieval cases."""

import pageindex as _pi
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from modules.registry import discover as _discover_modules, defaults as _module_defaults
from retrieval_cases import RETRIEVAL_CASES

from ..ollama_pool import ollama_bin, set_ollama_instances
from ..run_state import chat_phase

router = APIRouter()


@router.get("/api/tests")
def get_tests():
    return JSONResponse(RETRIEVAL_CASES)


@router.get("/api/status")
def get_status():
    activity = _pi.get_activity()

    return JSONResponse({
        "instances": [
            {"url": url, "index": i, "active": activity.get(url, 0)}
            for i, url in enumerate(_pi.OLLAMA_URLS)
        ],
        "any_busy": any(v > 0 for v in activity.values()),
        "progress": _pi.get_progress(),
        "live":     _pi.get_live_events(),
        "chat":     chat_phase(),
    })


@router.get("/api/config")
def get_config():
    return JSONResponse({
        "ollama_instances": len(_pi.OLLAMA_URLS),
        "ollama_urls": _pi.OLLAMA_URLS,
        "ollama_bin_available": ollama_bin() is not None,
        "retrieval_model": _pi.settings.model,
        "synthesis_model": _pi.settings.synthesis_model,
    })


class ConfigRequest(BaseModel):
    ollama_instances: int
    retrieval_model: Optional[str] = None
    synthesis_model: Optional[str] = None


@router.post("/api/config")
def post_config(req: ConfigRequest):
    result = set_ollama_instances(req.ollama_instances)
    if req.retrieval_model:
        _pi.settings.model = req.retrieval_model
    if req.synthesis_model:
        _pi.settings.synthesis_model = req.synthesis_model
    result["retrieval_model"] = _pi.settings.model
    result["synthesis_model"] = _pi.settings.synthesis_model
    return JSONResponse(result)


@router.get("/api/models")
def list_models():
    try:
        client = _pi.make_client(_pi.OLLAMA_URLS[0])
        res = client.list()
        models = [m["name"] for m in res.get("models", [])]
        return JSONResponse({"models": models})
    except Exception as e:
        return JSONResponse({"error": str(e), "models": []})


@router.get("/api/modules")
def get_modules():
    registry = _discover_modules()
    defs = _module_defaults()
    return JSONResponse({
        "stages": {
            stage: {
                "default": defs.get(stage),
                "modules": list(mods.values()),
            }
            for stage, mods in registry.items()
        }
    })
