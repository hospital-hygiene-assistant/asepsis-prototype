"""Running retrieval, and explaining why a node was not selected."""

import pageindex as _pi
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from modules.registry import defaults as _module_defaults
from pageindex.library import (
    ExpectedLibraryCorrupt,
    ExpectedLibraryNotBuilt,
    ExpectedLibraryStore,
)
from paths import LIBRARY_DIR
from retrieval_cases import RETRIEVAL_CASES

from ..answering_runtime import build_whole_library_retrieval
from ..ollama_pool import ensure_explainer
from ..runs import registry
from ..scoring import eval_case
router = APIRouter()


def _document_href(generation_id: str, document_id: str) -> str:
    return (
        f"/api/library/{generation_id}/documents/"
        f"{quote(document_id, safe='')}"
    )


def _bind_pin_asset(pin, base_href: str):
    if not isinstance(pin, dict) or not isinstance(pin.get("asset"), dict):
        return pin
    asset = dict(pin["asset"])
    asset_id = asset.get("id")
    if isinstance(asset_id, str):
        asset["image"] = f"{base_href}/assets/{quote(asset_id, safe='')}"
    return {**pin, "asset": asset}


def _bind_tree_assets(nodes: list[dict], base_href: str) -> None:
    for node in nodes:
        node["pin"] = _bind_pin_asset(node.get("pin"), base_href)
        _bind_tree_assets(node.get("children", []), base_href)


class ExplainRequest(BaseModel):
    generation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    stem: str
    node_id: str
    query: str


@router.post("/api/explain")
def explain_node(req: ExplainRequest):
    """On-demand grounded explanation for a node that was NOT selected.

    Runs on a dedicated Ollama instance so it never disturbs an in-flight query.
    """
    stem = Path(req.stem).name
    try:
        snapshot = ExpectedLibraryStore(LIBRARY_DIR).open_generation(
            req.generation_id
        )
        document = snapshot.document(stem)
    except KeyError:
        return JSONResponse(
            {"error": f"unknown document '{req.stem}'"}, status_code=404
        )
    except (ExpectedLibraryNotBuilt, ExpectedLibraryCorrupt):
        return JSONResponse(
            {"error": "Expected library unavailable"}, status_code=503
        )
    try:
        node = document.index.node(req.node_id)
    except KeyError:
        return JSONResponse({"error": f"unknown node '{req.node_id}'"}, status_code=404)

    url = ensure_explainer()
    if not url:
        return JSONResponse({"error": "no Ollama instance available"}, status_code=503)

    client = _pi.make_client(url)
    result = _pi.explain_nonselection(
        document.index, node.node_id, req.query, req.stem, client, url
    )
    result["node_id"] = req.node_id
    result["instance"] = url
    result["pin"] = _bind_pin_asset(
        node.pin.to_dict() if node.pin else None,
        _document_href(req.generation_id, stem),
    )
    return JSONResponse(result)


class RunRequest(BaseModel):
    # The client's own id for this run, so it can poll /api/runs/{run_id} from
    # the moment it sends this.
    run_id: Optional[str] = None
    test_id: Optional[str] = None
    query: Optional[str] = None
    ingest_module: Optional[str] = None


@router.post("/api/run")
def run_query(req: RunRequest):
    test = None
    query = req.query

    if req.test_id:
        test = next((t for t in RETRIEVAL_CASES if t["id"] == req.test_id), None)
        if test:
            query = test["query"]

    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)

    run = registry.create(req.run_id)
    run.begin_retrieval()
    try:
        retrieval = build_whole_library_retrieval()
    except Exception:
        run.fail()
        return JSONResponse(
            {"error": "retrieval runtime unavailable"}, status_code=503
        )
    try:
        search = retrieval.search(query, run)
    except Exception:
        run.fail()
        raise
    run.complete()
    results = search.to_debug_results()
    if search.generation_id is not None:
        for document_id, result in results.items():
            base_href = _document_href(search.generation_id, document_id)
            result.update({
                "generation_id": search.generation_id,
                "full_href": f"{base_href}/full",
                "page_href": f"{base_href}/page/{{page}}",
            })
            _bind_tree_assets(result.get("tree", []), base_href)
            for node in result.get("nodes", []):
                node["pin"] = _bind_pin_asset(node.get("pin"), base_href)

    test_result = eval_case(test, results) if test else None
    return JSONResponse({
        "run_id": run.id,
        "query": query,
        "results": results,
        "test_result": test_result,
        "coverage": {
            "status": search.status.value,
            "generation_id": search.generation_id,
            "diagnostics": [
                {
                    "document": item.document,
                    "node_id": item.node_id,
                    "code": item.code,
                    "message": item.message,
                }
                for item in search.diagnostics
            ],
        },
        "pipeline": {
            "ingest": req.ingest_module or _module_defaults()["ingest"],
        },
    })
