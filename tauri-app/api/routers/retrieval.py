"""Running retrieval, and explaining why a node was not selected."""

import pageindex as _pi
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

from modules.registry import load as _load_module, defaults as _module_defaults
from paths import INDEX_DIR
from retrieval_cases import RETRIEVAL_CASES

from ..ollama_pool import ensure_explainer
from ..retrieval import run_retrieval
from ..scoring import eval_case
from ..trees import read_tree

router = APIRouter()


class ExplainRequest(BaseModel):
    stem: str
    node_id: str
    query: str


@router.post("/api/explain")
def explain_node(req: ExplainRequest):
    """On-demand grounded explanation for a node that was NOT selected.

    Runs on a dedicated Ollama instance so it never disturbs an in-flight query.
    """
    index_mod = _load_module("index", _module_defaults()["index"])
    idx = INDEX_DIR / f"{Path(req.stem).name}.json"
    if not idx.exists():
        return JSONResponse({"error": f"unknown document '{req.stem}'"}, status_code=404)

    nodes = [_pi._node_from_dict(d) for d in read_tree(idx)]
    nodes_by_id = _pi._build_nodes_by_id(nodes)
    node = nodes_by_id.get(req.node_id)
    if node is None:
        return JSONResponse({"error": f"unknown node '{req.node_id}'"}, status_code=404)

    url = ensure_explainer()
    if not url:
        return JSONResponse({"error": "no Ollama instance available"}, status_code=503)

    parent_map = _pi._build_parent_map(nodes)
    breadcrumb = _pi._make_breadcrumb(req.node_id, parent_map, nodes_by_id)
    client = _pi.make_client(url)
    result = _pi.explain_nonselection(node, req.query, req.stem, breadcrumb, client, url)
    result["node_id"] = req.node_id
    result["instance"] = url
    result["pin"] = node.pin
    return JSONResponse(result)


class RunRequest(BaseModel):
    test_id: Optional[str] = None
    query: Optional[str] = None
    # Module selections — default to pipeline defaults when not supplied
    ingest_module: Optional[str] = None
    index_module:  Optional[str] = None
    query_module:  Optional[str] = None


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

    # Resolve modules
    defs = _module_defaults()
    index_mod_name = req.index_module or defs["index"]
    try:
        index_mod = _load_module("index", index_mod_name)
    except Exception as exc:
        return JSONResponse({"error": f"Could not load index module '{index_mod_name}': {exc}"}, status_code=400)

    results = run_retrieval(query, index_mod)

    test_result = eval_case(test, results) if test else None
    return JSONResponse({
        "query": query,
        "results": results,
        "test_result": test_result,
        "pipeline": {
            "ingest": req.ingest_module or _module_defaults()["ingest"],
            "index":  index_mod_name,
            "query":  req.query_module  or _module_defaults()["query"],
        },
    })
