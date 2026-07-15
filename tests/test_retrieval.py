"""End-to-end retrieval quality against the real index.

Needs Ollama running with the configured model, and a built index. Slow: every
case walks every document. Skips rather than fails when either is absent.

The cases live in retrieval_cases.py, shared with /api/tests in the dev console.

Run: pytest tests/test_retrieval.py -v
"""

import urllib.request

import pytest

from pageindex import OLLAMA_URLS, retrieve
from paths import INDEX_DIR
from retrieval_cases import RETRIEVAL_CASES


@pytest.fixture(scope="module", autouse=True)
def _model_must_be_reachable():
    """Skip when the model is down, rather than blaming the retriever.

    retrieve() returns [] whether it searched and found nothing or never ran, so
    with Ollama stopped every case here failed as "Expected nodes NOT retrieved"
    — a claim about retrieval quality, when in truth nothing was read. That is
    the same conflation of a technical fault with a finding that /api/chat had to
    be fixed for, and the suite should not repeat it. Drop this skip once the
    engine itself distinguishes the two (see C3 in tests/test_known_gaps.py).
    """
    probe = f"{OLLAMA_URLS[0].rstrip('/')}/api/tags"
    try:
        urllib.request.urlopen(probe, timeout=2).close()
    except OSError as exc:
        pytest.skip(
            f"Ollama unreachable at {OLLAMA_URLS[0]} ({exc}) — "
            f"no statement about retrieval quality can be made without it"
        )


def _retrieved_ids(doc_name: str, query: str) -> set[str]:
    try:
        return {node.node_id for node in retrieve(doc_name, query)}
    except FileNotFoundError:
        pytest.skip(f"Index for '{doc_name}' not built — run 'pipeline.py index' first")


def _all_retrieved(query: str) -> dict[str, set[str]]:
    found = {}
    for idx in sorted(INDEX_DIR.glob("*.json")):
        ids = _retrieved_ids(idx.stem, query)
        if ids:
            found[idx.stem] = ids
    return found


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
