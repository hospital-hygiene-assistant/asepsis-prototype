"""End-to-end retrieval quality against the real index.

Needs Ollama running with the configured model, and a built index. Slow: every
case walks every document. Skips rather than fails when either is absent.

The cases live in retrieval_cases.py, shared with /api/tests in the dev console.

Run: pytest tests/test_retrieval.py -v
"""

import urllib.request

import pytest

import pageindex
from pageindex import OLLAMA_URLS
from api.retrieval import LibraryStatus, WholeLibraryRetrieval
from api.runs import RunRegistry
from retrieval_cases import RETRIEVAL_CASES


@pytest.fixture(scope="module", autouse=True)
def _model_must_be_reachable():
    """Skip when the model is down, rather than blaming the retriever.

    A model outage is setup failure, not a retrieval-quality finding.
    """
    probe = f"{OLLAMA_URLS[0].rstrip('/')}/api/tags"
    try:
        urllib.request.urlopen(probe, timeout=2).close()
    except OSError as exc:
        pytest.skip(
            f"Ollama unreachable at {OLLAMA_URLS[0]} ({exc}) — "
            f"no statement about retrieval quality can be made without it"
        )


def _all_retrieved(query: str) -> dict[str, set[str]]:
    try:
        result = WholeLibraryRetrieval(pageindex).search(
            query, RunRegistry().create()
        )
    except (FileNotFoundError, ValueError, KeyError):
        pytest.skip("Index generation unavailable — run 'pipeline.py index' first")
    if result.status is LibraryStatus.UNAVAILABLE:
        pytest.skip("Library search unavailable; no quality finding can be made")
    assert result.status is LibraryStatus.COMPLETE, (
        "Retrieval quality cannot be scored from a partial search: "
        f"{result.diagnostics}"
    )
    return {
        document.document: {item.node.node_id for item in document.evidence}
        for document in result.documents
        if document.evidence
    }


def _describe(retrieved: dict[str, set[str]]) -> str:
    return "\n".join(f"  [{doc}] {sorted(ids)}" for doc, ids in retrieved.items()) or "  (nothing)"


@pytest.mark.parametrize("case", RETRIEVAL_CASES, ids=lambda c: c["id"])
def test_retrieval_case(case):
    retrieved = _all_retrieved(case["query"])

    missing = {
        doc: set(ids) - retrieved.get(doc, set())
        for doc, ids in case["expected"].items()
        if set(ids) - retrieved.get(doc, set())
    }
    assert not missing, (
        f"{case['category']} | {case['description']}\n"
        f"Expected nodes NOT retrieved:\n"
        + "\n".join(f"  [{doc}] {sorted(ids)}" for doc, ids in missing.items())
        + f"\nActually retrieved:\n{_describe(retrieved)}"
    )

    for doc, ids in case["expected_any"].items():
        got = retrieved.get(doc, set())
        assert got & set(ids), (
            f"{case['category']} | {case['description']}\n"
            f"Expected at least one of {sorted(ids)} from [{doc}], got: {sorted(got)}"
        )

    spurious = {
        doc: set(ids) & retrieved.get(doc, set())
        for doc, ids in case["forbidden"].items()
        if set(ids) & retrieved.get(doc, set())
    }
    assert not spurious, (
        f"{case['category']} | {case['description']}\n"
        f"Forbidden nodes WERE retrieved:\n"
        + "\n".join(f"  [{doc}] {sorted(ids)}" for doc, ids in spurious.items())
    )
