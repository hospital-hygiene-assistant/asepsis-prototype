"""PageIndex Explorer — FastAPI backend. Serves the UI and wraps pageindex retrieval."""

import json
import sys
from pathlib import Path
from typing import Optional

# Add project root so we can import pageindex and modules
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pageindex import _node_from_dict
from modules.registry import discover as _discover_modules, load as _load_module, defaults as _module_defaults

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="PageIndex Explorer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UI_DIR = Path(__file__).parent / "ui"
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

# ---------------------------------------------------------------------------
# Test definitions — mirrors tests/test_retrieval.py exactly
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": "test_sodium_restriction",
        "category": "SINGLE",
        "description": "Sodium restriction",
        "query": "What sodium intake level is recommended for hypertension and by how much does it reduce blood pressure?",
        "expected": {"hypertension_guidelines": ["sodium-restriction"]},
        "expected_any": {},
        "forbidden": {"hypertension_guidelines": ["first-line-drug-classes", "fourth-line-agents"]},
    },
    {
        "id": "test_nmba_icu_two_leaves",
        "category": "SINGLE",
        "description": "NMBA in ARDS — two leaves",
        "query": "When are neuromuscular blocking agents indicated in ARDS patients and how is the depth of blockade monitored?",
        "expected": {"icu_sedation_guide": ["indications-in-ards", "monitoring-and-safety"]},
        "expected_any": {},
        "forbidden": {"icu_sedation_guide": ["propofol", "dexmedetomidine", "benzodiazepines"]},
    },
    {
        "id": "test_hypertension_lifestyle_and_drugs",
        "category": "MULTI",
        "description": "Lifestyle + drug classes",
        "query": "What lifestyle changes and which drug classes should be started for newly diagnosed hypertension?",
        "expected": {"hypertension_guidelines": ["first-line-drug-classes"]},
        "expected_any": {
            "hypertension_guidelines": [
                "sodium-restriction", "dash-diet",
                "exercise-and-weight-management", "non-pharmacological-management-overview",
            ],
        },
        "forbidden": {},
    },
    {
        "id": "test_sepsis_antibiotics_empiric_and_deescalation",
        "category": "MULTI",
        "description": "Sepsis antibiotics — empiric + de-escalation",
        "query": "How should empiric antibiotics be chosen for sepsis by source of infection, and when should they be narrowed?",
        "expected": {"antibiotic_stewardship": ["empiric-regimens-by-source", "de-escalation-and-duration"]},
        "expected_any": {},
        "forbidden": {},
    },
    {
        "id": "test_hypertension_ckd_cross_doc",
        "category": "CROSS",
        "description": "CKD antihypertensives + renal screening",
        "query": "What antihypertensives are preferred for patients with CKD and what renal complications should be monitored?",
        "expected": {
            "hypertension_guidelines": ["hypertension-in-ckd"],
            "diabetes_management": ["complication-screening"],
        },
        "expected_any": {},
        "forbidden": {},
    },
    {
        "id": "test_septic_icu_patient",
        "category": "CROSS",
        "description": "Septic ICU patient — antibiotics + sedation",
        "query": "A patient with septic shock is intubated in the ICU — what empiric antibiotics and sedation agents should be used?",
        "expected": {"antibiotic_stewardship": ["empiric-regimens-by-source"]},
        "expected_any": {"icu_sedation_guide": ["opioids", "propofol", "dexmedetomidine"]},
        "forbidden": {},
    },
]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/tests")
def get_tests():
    return JSONResponse(TEST_CASES)


@app.get("/api/modules")
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


@app.get("/api/documents")
def get_documents():
    # Use the default index module to find where indices live
    index_mod = _load_module("index", _module_defaults()["index"])
    index_dir = getattr(index_mod, "INDEX_DIR", Path("index"))
    docs = []
    for idx in sorted(index_dir.glob("*.json")):
        tree = json.loads(idx.read_text(encoding="utf-8"))
        nodes = [_node_from_dict(d) for d in tree]
        docs.append({
            "name": idx.stem,
            "leaf_count": _count_leaves(nodes),
            "tree": tree,
        })
    return JSONResponse(docs)


class RunRequest(BaseModel):
    test_id: Optional[str] = None
    query: Optional[str] = None
    # Module selections — default to pipeline defaults when not supplied
    ingest_module: Optional[str] = None
    index_module:  Optional[str] = None
    query_module:  Optional[str] = None


@app.post("/api/run")
def run_query(req: RunRequest):
    test = None
    query = req.query

    if req.test_id:
        test = next((t for t in TEST_CASES if t["id"] == req.test_id), None)
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

    index_dir = getattr(index_mod, "INDEX_DIR", Path("index"))

    results = {}
    for idx in sorted(index_dir.glob("*.json")):
        doc_name = idx.stem
        try:
            nodes = index_mod.retrieve(doc_name, query)
        except FileNotFoundError:
            continue

        raw_tree = json.loads(idx.read_text(encoding="utf-8"))
        results[doc_name] = {
            "tree": raw_tree,
            "retrieved_ids": [n.node_id for n in nodes],
            "nodes": [
                {
                    "node_id": n.node_id,
                    "title": n.title,
                    "content": n.content or "",
                    "synthetic": n.synthetic,
                    "heading_level": n.heading_level,
                    "summary": n.summary,
                }
                for n in nodes
            ],
        }

    test_result = _eval_test(test, results) if test else None
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_leaves(nodes):
    count = 0
    for n in nodes:
        if n.is_leaf:
            count += 1
        else:
            count += _count_leaves(n.children)
    return count


def _eval_test(test, results):
    expected = test.get("expected", {})
    expected_any = test.get("expected_any", {})
    forbidden = test.get("forbidden", {})

    missing = {
        doc: list(set(ids) - set(results.get(doc, {}).get("retrieved_ids", [])))
        for doc, ids in expected.items()
        if set(ids) - set(results.get(doc, {}).get("retrieved_ids", []))
    }
    any_missing = {
        doc: ids
        for doc, ids in expected_any.items()
        if not (set(ids) & set(results.get(doc, {}).get("retrieved_ids", [])))
    }
    spurious = {
        doc: list(set(ids) & set(results.get(doc, {}).get("retrieved_ids", [])))
        for doc, ids in forbidden.items()
        if set(ids) & set(results.get(doc, {}).get("retrieved_ids", []))
    }

    return {
        "passed": not missing and not any_missing and not spurious,
        "missing": missing,
        "any_missing": any_missing,
        "spurious": spurious,
        "expected": expected,
        "expected_any": expected_any,
        "forbidden": forbidden,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
