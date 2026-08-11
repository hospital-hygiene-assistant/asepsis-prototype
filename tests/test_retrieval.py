"""
End-to-end retrieval tests for the PageIndex pipeline.
Requires Ollama running with the configured model. Slow (~2 min total).

Each test defines:
  expected  — {doc_name: set of nodeIds} that MUST appear in results
  forbidden — {doc_name: set of nodeIds} that must NOT appear (optional)

Categories:
  SINGLE    — narrow query relevant to one document, one specific leaf
  MULTI     — query spanning multiple sections of the same document
  CROSS     — composite query requiring nodes from two different documents

Run: pytest tests/test_retrieval.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pageindex import retrieve, INDEX_DIR

# This module is the live tier: it needs a running Ollama AND a built index.
# It is deselected from the default offline run and included via
#   python3 run_tests.py --llm
pytestmark = pytest.mark.llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_index() -> None:
    if not INDEX_DIR.exists() or not any(INDEX_DIR.glob("*.json")):
        pytest.skip("No index built — run `python3 pipeline.py index` first")


def _retrieved_ids(doc_name: str, query: str) -> set[str]:
    try:
        nodes = retrieve(doc_name, query)
    except FileNotFoundError:
        pytest.skip(f"Index for '{doc_name}' not built — run pageindex.py first")
    return {n.node_id for n in nodes}


def _all_retrieved(query: str) -> dict[str, set[str]]:
    _require_index()
    result: dict[str, set[str]] = {}
    for idx in sorted(INDEX_DIR.glob("*.json")):
        ids = _retrieved_ids(idx.stem, query)
        if ids:
            result[idx.stem] = ids
    return result


def _assert_retrieved(retrieved: dict[str, set[str]], expected: dict[str, set[str]]):
    missing = {
        doc: ids - retrieved.get(doc, set())
        for doc, ids in expected.items()
        if ids - retrieved.get(doc, set())
    }
    assert not missing, (
        "Expected nodes NOT retrieved:\n"
        + "\n".join(f"  [{d}] {ids}" for d, ids in missing.items())
        + "\n\nActual:\n"
        + "\n".join(f"  [{d}] {ids}" for d, ids in retrieved.items())
    )


def _assert_not_retrieved(retrieved: dict[str, set[str]], forbidden: dict[str, set[str]]):
    spurious = {
        doc: ids & retrieved.get(doc, set())
        for doc, ids in forbidden.items()
        if ids & retrieved.get(doc, set())
    }
    assert not spurious, (
        "Forbidden nodes WERE retrieved:\n"
        + "\n".join(f"  [{d}] {ids}" for d, ids in spurious.items())
    )


# ---------------------------------------------------------------------------
# SINGLE-DOCUMENT
# ---------------------------------------------------------------------------

class TestSingleDocument:

    def test_sodium_restriction(self):
        """
        SINGLE | hypertension_guidelines
        A specific sodium/diet query should return sodium-restriction and NOT drug nodes.
        """
        q = "What sodium intake level is recommended for hypertension and by how much does it reduce blood pressure?"
        retrieved = _all_retrieved(q)
        _assert_retrieved(retrieved, {
            "hypertension_guidelines": {"sodium-restriction"},
        })
        _assert_not_retrieved(retrieved, {
            "hypertension_guidelines": {"first-line-drug-classes", "fourth-line-agents"},
        })

    def test_nmba_icu_two_leaves(self):
        """
        SINGLE | icu_sedation_guide
        Neuromuscular blockade question spanning two leaves in the same section.
        Both indications and monitoring must be retrieved.
        """
        q = "When are neuromuscular blocking agents indicated in ARDS patients and how is the depth of blockade monitored?"
        retrieved = _all_retrieved(q)
        _assert_retrieved(retrieved, {
            "icu_sedation_guide": {"indications-in-ards", "monitoring-and-safety"},
        })
        _assert_not_retrieved(retrieved, {
            "icu_sedation_guide": {"propofol", "dexmedetomidine", "benzodiazepines"},
        })


# ---------------------------------------------------------------------------
# MULTI-SECTION SAME DOCUMENT
# ---------------------------------------------------------------------------

class TestMultiSectionSameDocument:

    def test_hypertension_lifestyle_and_drugs(self):
        """
        MULTI | hypertension_guidelines
        Asking about both lifestyle and drugs requires nodes from two different
        top-level sections of the same document.
        """
        q = "What lifestyle changes and which drug classes should be started for newly diagnosed hypertension?"
        retrieved = _all_retrieved(q)
        _assert_retrieved(retrieved, {
            "hypertension_guidelines": {"first-line-drug-classes"},
        })
        # At least one lifestyle leaf must also be present
        lifestyle_ids = {"sodium-restriction", "dash-diet", "exercise-and-weight-management",
                         "non-pharmacological-management-overview"}
        got = retrieved.get("hypertension_guidelines", set())
        assert got & lifestyle_ids, (
            f"Expected at least one lifestyle node, got: {got}"
        )

    def test_sepsis_antibiotics_empiric_and_deescalation(self):
        """
        MULTI | antibiotic_stewardship
        Sepsis antibiotic management spans two sibling leaves in the same section.
        """
        q = "How should empiric antibiotics be chosen for sepsis by source of infection, and when should they be narrowed?"
        retrieved = _all_retrieved(q)
        _assert_retrieved(retrieved, {
            "antibiotic_stewardship": {
                "empiric-regimens-by-source",
                "de-escalation-and-duration",
            },
        })


# ---------------------------------------------------------------------------
# CROSS-DOCUMENT
# ---------------------------------------------------------------------------

class TestCrossDocument:

    def test_hypertension_ckd_cross_doc(self):
        """
        CROSS | hypertension_guidelines + diabetes_management
        CKD-specific antihypertensives are in hypertension doc;
        renal complication screening details are in diabetes doc.
        """
        q = "What antihypertensives are preferred for patients with CKD and what renal complications should be monitored?"
        retrieved = _all_retrieved(q)
        _assert_retrieved(retrieved, {
            "hypertension_guidelines": {"hypertension-in-ckd"},
            "diabetes_management": {"complication-screening"},
        })

    def test_septic_icu_patient(self):
        """
        CROSS | antibiotic_stewardship + icu_sedation_guide
        A septic intubated patient requires empiric antibiotics (stewardship doc)
        and sedation management (ICU doc). Both must be retrieved.
        """
        q = "A patient with septic shock is intubated in the ICU — what empiric antibiotics and sedation agents should be used?"
        retrieved = _all_retrieved(q)
        _assert_retrieved(retrieved, {
            "antibiotic_stewardship": {"empiric-regimens-by-source"},
        })
        # At least one sedative agent leaf required from ICU doc
        sedative_ids = {"opioids", "propofol", "dexmedetomidine"}
        got = retrieved.get("icu_sedation_guide", set())
        assert got & sedative_ids, (
            f"Expected at least one sedative node from icu_sedation_guide, got: {got}"
        )
