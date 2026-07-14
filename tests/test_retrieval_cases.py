"""The console and the pytest suite must judge a retrieval case the same way.

They used to hold separate copies of the cases and check different things, so a
case could pass in one and fail in the other.
"""


import pytest


from retrieval_cases import RETRIEVAL_CASES
from api.scoring import eval_case


def results_with(**by_doc):
    """Shape a retrieval result the way _eval_test reads it."""
    return {doc: {"retrieved_ids": ids} for doc, ids in by_doc.items()}


class TestCaseData:
    def test_every_case_declares_all_three_assertions(self):
        for case in RETRIEVAL_CASES:
            assert set(case) >= {"id", "category", "description", "query",
                                 "expected", "expected_any", "forbidden"}, case["id"]

    def test_ids_are_unique(self):
        ids = [c["id"] for c in RETRIEVAL_CASES]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("case", RETRIEVAL_CASES, ids=lambda c: c["id"])
    def test_expected_and_forbidden_do_not_contradict(self, case):
        # A case demanding and banning the same node can never pass.
        for doc, ids in case["forbidden"].items():
            assert not set(ids) & set(case["expected"].get(doc, [])), doc
            assert not set(ids) & set(case["expected_any"].get(doc, [])), doc


class TestEvalTest:
    def test_passes_when_every_expectation_holds(self):
        case = {"expected": {"doc": ["a"]}, "expected_any": {}, "forbidden": {}}
        assert eval_case(case, results_with(doc=["a", "b"]))["passed"]

    def test_fails_on_a_missing_required_node(self):
        case = {"expected": {"doc": ["a"]}, "expected_any": {}, "forbidden": {}}
        verdict = eval_case(case, results_with(doc=["b"]))
        assert not verdict["passed"]
        assert verdict["missing"] == {"doc": ["a"]}

    def test_fails_when_no_any_of_node_is_present(self):
        case = {"expected": {}, "expected_any": {"doc": ["a", "b"]}, "forbidden": {}}
        assert not eval_case(case, results_with(doc=["z"]))["passed"]

    def test_one_any_of_node_is_enough(self):
        case = {"expected": {}, "expected_any": {"doc": ["a", "b"]}, "forbidden": {}}
        assert eval_case(case, results_with(doc=["b"]))["passed"]

    def test_fails_when_a_forbidden_node_is_retrieved(self):
        # The console ignored forbidden entirely, so a case the suite failed
        # could still show a green tick.
        case = {"expected": {"doc": ["a"]}, "expected_any": {}, "forbidden": {"doc": ["x"]}}
        verdict = eval_case(case, results_with(doc=["a", "x"]))
        assert not verdict["passed"]
        assert verdict["spurious"] == {"doc": ["x"]}

    def test_forbidden_node_absent_is_a_pass(self):
        case = {"expected": {"doc": ["a"]}, "expected_any": {}, "forbidden": {"doc": ["x"]}}
        assert eval_case(case, results_with(doc=["a"]))["passed"]

    def test_a_document_that_returned_nothing_is_not_an_error(self):
        case = {"expected": {"doc": ["a"]}, "expected_any": {}, "forbidden": {}}
        verdict = eval_case(case, {})
        assert not verdict["passed"]
        assert verdict["missing"] == {"doc": ["a"]}
