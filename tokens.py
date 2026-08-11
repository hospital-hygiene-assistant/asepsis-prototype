"""
Token estimation and budgeting.

Ollama exposes no tokenizer endpoint, so exact counts are unavailable before a
call. We estimate from character length and then *calibrate*: every response
carries `prompt_eval_count`, the real token count of the prompt we just sent.
Feeding that back gives a measured chars-per-token ratio instead of a guessed
one, and the error is observable via `calibration()` rather than assumed.

The estimator deliberately errs high. Under-estimating means overflowing the
model's window and silently losing passages; over-estimating only means
leaving a little context unused.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

# Conservative starting point: English prose runs ~4 chars/token, so dividing
# by a smaller number over-estimates. Markdown, code and tables run denser.
_INITIAL_CHARS_PER_TOKEN = 3.4

# Calibration never strays outside this band — a few pathological samples
# should not be able to make the estimator wildly optimistic.
_MIN_RATIO, _MAX_RATIO = 2.0, 5.0

_lock = threading.Lock()
_ratio = _INITIAL_CHARS_PER_TOKEN
_samples = 0
_abs_error_sum = 0.0


def estimate_tokens(text: str | None) -> int:
    """Estimated token count for `text`, biased to over-estimate."""
    if not text:
        return 0
    with _lock:
        ratio = _ratio
    return max(1, int(len(text) / ratio) + 1)


def observe(prompt: str, actual_tokens: int) -> None:
    """Feed a real `prompt_eval_count` back into the estimator.

    Called after every LLM response that reports one. The ratio converges on
    the true chars-per-token for this corpus and model.
    """
    global _ratio, _samples, _abs_error_sum
    if not prompt or not actual_tokens or actual_tokens <= 0:
        return
    observed = len(prompt) / actual_tokens
    with _lock:
        estimated = max(1, int(len(prompt) / _ratio) + 1)
        _abs_error_sum += abs(estimated - actual_tokens) / actual_tokens
        _samples += 1
        # Exponential moving average, then bias ~7% toward over-estimating.
        alpha = 0.2
        blended = (1 - alpha) * _ratio + alpha * observed
        _ratio = max(_MIN_RATIO, min(_MAX_RATIO, blended * 0.93))


def calibration() -> dict:
    """Current estimator state, surfaced in /api/status."""
    with _lock:
        return {
            "chars_per_token": round(_ratio, 3),
            "samples": _samples,
            "mean_abs_error": round(_abs_error_sum / _samples, 4) if _samples else None,
        }


def reset_for_tests() -> None:
    global _ratio, _samples, _abs_error_sum
    with _lock:
        _ratio, _samples, _abs_error_sum = _INITIAL_CHARS_PER_TOKEN, 0, 0.0


@dataclass
class BudgetResult:
    fitted: list          # items that fit, in the order given
    dropped: list         # items that did not, in the order given
    tokens_used: int
    tokens_max: int

    @property
    def did_overflow(self) -> bool:
        return bool(self.dropped)


def budget_fit(items: list, max_tokens: int, cost, *, overhead: int = 0) -> BudgetResult:
    """Greedily accept `items` in order until the budget is exhausted.

    `cost(item) -> int` gives an item's token cost. Accepting stops at the
    FIRST item that does not fit — later, cheaper items are not squeezed in.
    That keeps the result a prefix of the input order, which matters when the
    order is a relevance ranking: silently preferring a short low-ranked
    passage over a long high-ranked one would misrepresent the ranking.
    """
    budget = max_tokens - overhead
    used = 0
    fitted, dropped = [], []
    for i, item in enumerate(items):
        if dropped:
            dropped.append(item)
            continue
        c = cost(item)
        if used + c > budget:
            dropped.append(item)
            continue
        used += c
        fitted.append(item)
    return BudgetResult(fitted=fitted, dropped=dropped,
                        tokens_used=used, tokens_max=max(0, budget))
