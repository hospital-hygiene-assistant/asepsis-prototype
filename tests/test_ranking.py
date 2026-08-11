"""
Phase 3 — BM25 ranking.

The ranking decides which candidates get evaluated before the agent's context
budget runs out, so what matters as much as relevance is DETERMINISM: a cached
run and a live run must agree, which means ties can never be resolved by dict
or thread ordering.
"""
import pytest

import ranking

pytestmark = pytest.mark.unit


CORPUS = [
    "Sodium restriction lowers blood pressure in hypertensive patients.",
    "Empiric antibiotics for sepsis should be chosen by source of infection.",
    "Propofol and dexmedetomidine are used for sedation in the ICU.",
    "Blood pressure targets differ for patients with chronic kidney disease.",
    "The committee met on Tuesday to review the guideline drafting process.",
]


def _rank(query, corpus=None):
    return ranking.rank(list(corpus or CORPUS), query, text_of=lambda s: s)


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert ranking.tokenize("Blood Pressure, 140/90!") == ["blood", "pressure", "140", "90"]

    def test_drops_stopwords_and_single_chars(self):
        assert ranking.tokenize("the a of x sodium") == ["sodium"]

    def test_empty_input(self):
        assert ranking.tokenize("") == []
        assert ranking.tokenize(None) == []


class TestRelevance:
    def test_best_match_ranks_first(self):
        top = _rank("sodium restriction blood pressure")[0]
        assert top.item.startswith("Sodium restriction")

    def test_unrelated_document_ranks_last(self):
        result = _rank("sodium restriction blood pressure")
        assert "committee met" in result[-1].item

    def test_every_item_is_returned_exactly_once(self):
        result = _rank("sepsis")
        assert len(result) == len(CORPUS)
        assert {s.item for s in result} == set(CORPUS)

    def test_ranks_are_sequential(self):
        assert [s.rank for s in _rank("sepsis")] == list(range(len(CORPUS)))

    def test_scores_are_never_negative(self):
        """A term present in every document must contribute ~0, not push the
        score below zero the way textbook idf does."""
        corpus = ["blood pressure high", "blood pressure low", "blood pressure normal"]
        assert all(s.score >= 0 for s in _rank("blood", corpus))


class TestDeterminism:
    def test_repeated_ranking_is_identical(self):
        first = [(s.item, s.rank) for s in _rank("sepsis antibiotics")]
        for _ in range(5):
            assert [(s.item, s.rank) for s in _rank("sepsis antibiotics")] == first

    def test_ties_keep_input_order(self):
        corpus = ["alpha text", "beta text", "gamma text"]
        result = _rank("nothing matches here", corpus)
        assert all(s.score == 0 for s in result)
        assert [s.item for s in result] == corpus, "ties must preserve input order"

    def test_input_order_change_does_not_reorder_scored_items(self):
        a = _rank("sepsis antibiotics")
        b = ranking.rank(list(reversed(CORPUS)), "sepsis antibiotics", text_of=lambda s: s)
        # The top scorer is unambiguous, so it must be first either way.
        assert a[0].item == b[0].item


class TestEdgeCases:
    def test_empty_corpus(self):
        assert ranking.rank([], "query", text_of=str) == []

    def test_empty_query_scores_zero(self):
        assert all(s.score == 0 for s in _rank(""))

    def test_single_document(self):
        result = _rank("sodium", ["Sodium restriction helps."])
        assert len(result) == 1 and result[0].rank == 0

    def test_documents_with_no_text(self):
        result = _rank("sodium", ["", "sodium restriction", ""])
        assert result[0].item == "sodium restriction"
