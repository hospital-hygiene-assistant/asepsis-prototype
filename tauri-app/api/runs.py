"""Retrieval runs, addressable by id.

A run is created by whoever starts it and polled by that client alone. Keying
them apart is what lets two clients retrieve at once: a shared run would mean
the second reset the first's progress, and both read a blend of the two.

Ids come from the client so it can poll from the moment it sends the request,
without waiting for a reply to learn what to poll.
"""

import threading
import time
from pageindex import QuestionRun

# A run is only interesting while its client watches it. These bounds stop a
# long-lived process accumulating every run it has ever served.
TTL_SECONDS = 30 * 60
MAX_RUNS = 64


class RunRegistry:
    """The runs this process is tracking."""

    def __init__(self, ttl: float = TTL_SECONDS, max_runs: int = MAX_RUNS) -> None:
        self._runs: dict[str, QuestionRun] = {}
        self._lock = threading.Lock()
        self._ttl = ttl
        self._max = max_runs

    def create(self, run_id: str | None = None) -> QuestionRun:
        run = QuestionRun(run_id)
        with self._lock:
            self._evict()
            self._runs[run.id] = run
        return run

    def get(self, run_id: str) -> QuestionRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def _evict(self) -> None:
        """Drop expired runs, then the oldest if still over the cap."""
        cutoff = time.monotonic() - self._ttl
        for run_id in [i for i, r in self._runs.items() if r.started < cutoff]:
            del self._runs[run_id]
        while len(self._runs) >= self._max:
            oldest = min(self._runs, key=lambda i: self._runs[i].started)
            del self._runs[oldest]
registry = RunRegistry()
