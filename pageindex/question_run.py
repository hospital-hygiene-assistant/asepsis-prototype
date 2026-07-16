"""Thread-safe lifecycle and retrieval audit for one question."""

import threading
import time
import uuid


class QuestionRun:
    """The complete mutable state of one question, safe for worker polling."""

    def __init__(self, run_id: str | None = None) -> None:
        self.id = run_id or uuid.uuid4().hex
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._phase = "idle"
        self._detail = ""
        self._total = 0
        self._done = 0
        self._pruned: set[str] = set()
        self._retrieved: set[str] = set()
        self._kept: set[str] = set()
        self._rejected: set[str] = set()
        self._errored: set[str] = set()
        self._meta: dict[str, dict] = {}

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def begin_retrieval(self, detail: str = "") -> None:
        self._set_phase("retrieval", detail)

    def set_total(self, total_leaves: int) -> None:
        if total_leaves < 0:
            raise ValueError("retrieval total cannot be negative")
        with self._lock:
            self._total = total_leaves
            self._done = 0
            for group in (
                self._pruned,
                self._retrieved,
                self._kept,
                self._rejected,
                self._errored,
            ):
                group.clear()
            self._meta.clear()

    def mark(self, status: str, node_id: str) -> None:
        groups = {
            "pruned": self._pruned,
            "retrieved": self._retrieved,
            "kept": self._kept,
            "rejected": self._rejected,
            "errored": self._errored,
        }
        try:
            group = groups[status]
        except KeyError as exc:
            raise ValueError(f"unknown retrieval verdict: {status}") from exc
        with self._lock:
            group.add(node_id)

    def record(
        self,
        node_id: str,
        status: str,
        reason: str = "",
        quote: str = "",
    ) -> None:
        with self._lock:
            self._meta[node_id] = {
                "status": status,
                "reason": reason,
                "quote": quote,
            }

    def leaf_complete(self) -> None:
        with self._lock:
            if self._done >= self._total:
                raise RuntimeError("retrieval completed more leaves than declared")
            self._done += 1

    def begin_synthesis(self, detail: str = "") -> None:
        self._set_phase("synthesis", detail)

    def complete(self, detail: str = "") -> None:
        self._set_phase("idle", detail)

    def fail(self, detail: str = "") -> None:
        self._set_phase("error", detail)

    def progress(self) -> dict[str, int]:
        with self._lock:
            return {"total": self._total, "done": self._done}

    def events(self) -> dict:
        with self._lock:
            return self._events_unlocked()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "run_id": self.id,
                "phase": self._phase,
                "detail": self._detail,
                "progress": {"total": self._total, "done": self._done},
                "live": self._events_unlocked(),
            }

    def _events_unlocked(self) -> dict:
        return {
            "pruned": list(self._pruned),
            "retrieved": list(self._retrieved),
            "kept": list(self._kept),
            "rejected": list(self._rejected),
            "errored": list(self._errored),
            "meta": {key: dict(value) for key, value in self._meta.items()},
        }

    def _set_phase(self, phase: str, detail: str) -> None:
        with self._lock:
            self._phase = phase
            self._detail = detail
