"""Scoring a retrieval case, to the same standard as the pytest suite."""


def eval_case(test, results):
    """Score one retrieval case, to the same standard as the pytest suite."""
    expected = test.get("expected", {})
    expected_any = test.get("expected_any", {})
    forbidden = test.get("forbidden", {})

    def retrieved(doc: str) -> set:
        return set(results.get(doc, {}).get("retrieved_ids", []))

    missing = {
        doc: sorted(set(ids) - retrieved(doc))
        for doc, ids in expected.items()
        if set(ids) - retrieved(doc)
    }
    any_missing = {
        doc: ids for doc, ids in expected_any.items() if not set(ids) & retrieved(doc)
    }
    spurious = {
        doc: sorted(set(ids) & retrieved(doc))
        for doc, ids in forbidden.items()
        if set(ids) & retrieved(doc)
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
