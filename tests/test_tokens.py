"""
Phase 0 — token estimation and budgeting.

The estimator must err HIGH: under-estimating overflows the model's window and
loses passages silently, over-estimating merely leaves context unused.
"""
import pytest

import tokens as app_tokens

pytestmark = pytest.mark.unit


class TestEstimator:
    def test_empty_is_zero(self):
        assert app_tokens.estimate_tokens("") == 0
        assert app_tokens.estimate_tokens(None) == 0

    def test_monotonic_in_length(self):
        prev = 0
        for n in (1, 10, 100, 1000, 10000):
            cur = app_tokens.estimate_tokens("a" * n)
            assert cur > prev
            prev = cur

    @pytest.mark.parametrize("text", [
        "The patient was started on amoxicillin 500mg three times daily.",
        "## Heading\n\n- bullet one\n- bullet two\n\n| a | b |\n|---|---|\n",
        "Ω≈ç√∫˜µ≤≥÷ éèêë 日本語テキスト",
    ])
    def test_over_estimates_real_text(self, text):
        # ~4 chars/token is the realistic figure for English prose; the
        # estimator should sit above it so budgets stay safe.
        realistic = len(text) / 4
        assert app_tokens.estimate_tokens(text) >= realistic

    def test_calibration_moves_the_ratio_toward_observed(self):
        prompt = "x" * 4000
        before = app_tokens.calibration()["chars_per_token"]
        for _ in range(20):
            app_tokens.observe(prompt, 800)      # observed 5.0 chars/token
        after = app_tokens.calibration()["chars_per_token"]
        assert after > before, "ratio should rise toward the observed 5.0"
        assert after <= 5.0, "but never past the safety ceiling"

    def test_calibration_is_bounded(self):
        for _ in range(50):
            app_tokens.observe("x" * 1000, 1000)   # absurd 1.0 chars/token
        assert app_tokens.calibration()["chars_per_token"] >= app_tokens._MIN_RATIO

    def test_observe_ignores_garbage(self):
        app_tokens.observe("", 100)
        app_tokens.observe("text", 0)
        app_tokens.observe("text", -5)
        assert app_tokens.calibration()["samples"] == 0

    def test_calibration_reports_error(self):
        app_tokens.observe("x" * 400, 100)
        cal = app_tokens.calibration()
        assert cal["samples"] == 1
        assert cal["mean_abs_error"] is not None


class TestBudgetFit:
    @staticmethod
    def cost(item):
        return item

    def test_everything_fits(self):
        r = app_tokens.budget_fit([10, 20, 30], 100, self.cost)
        assert r.fitted == [10, 20, 30]
        assert r.dropped == []
        assert r.tokens_used == 60
        assert not r.did_overflow

    def test_empty_input(self):
        r = app_tokens.budget_fit([], 100, self.cost)
        assert r.fitted == [] and r.dropped == []

    def test_single_oversized_item_is_dropped_not_crashed(self):
        r = app_tokens.budget_fit([500], 100, self.cost)
        assert r.fitted == [] and r.dropped == [500]

    def test_exact_fit_is_accepted(self):
        r = app_tokens.budget_fit([60, 40], 100, self.cost)
        assert r.fitted == [60, 40] and r.tokens_used == 100

    def test_one_over_is_rejected(self):
        r = app_tokens.budget_fit([60, 41], 100, self.cost)
        assert r.fitted == [60] and r.dropped == [41]

    def test_result_is_a_prefix_of_input_order(self):
        """A ranking must not be reordered: once something doesn't fit, we stop.
        Squeezing in a later, cheaper item would misrepresent the ranking."""
        r = app_tokens.budget_fit([90, 50, 5], 100, self.cost)
        assert r.fitted == [90]
        assert r.dropped == [50, 5], "the cheap item must NOT jump the queue"

    def test_overhead_reduces_the_budget(self):
        r = app_tokens.budget_fit([60], 100, self.cost, overhead=50)
        assert r.fitted == [] and r.tokens_max == 50
