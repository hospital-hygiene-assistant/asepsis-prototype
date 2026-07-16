"""Running retrieval, and explaining why a node was not selected."""

import pageindex as _pi
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

from modules.registry import defaults as _module_defaults
from pageindex.generations import IndexGenerationStore
from paths import INDEX_DIR
from retrieval_cases import RETRIEVAL_CASES

from ..ollama_pool import ensure_explainer
from ..retrieval import WholeLibraryRetrieval
from ..runs import registry
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
    stem = Path(req.stem).name
    with IndexGenerationStore(INDEX_DIR).pin_current() as snapshot:
        idx = next((path for path in snapshot.document_paths if path.stem == stem), None)
        if idx is None:
            return JSONResponse(
                {"error": f"unknown document '{req.stem}'"}, status_code=404
            )
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
    result["pin"] = node.pin.to_dict() if node.pin else None
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
    search = WholeLibraryRetrieval(_pi).search(query, run)
    results = search.to_debug_results()

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
