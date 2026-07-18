"""Thread-safe lifecycle and retrieval audit for one question."""

import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum


class QuestionCancelled(RuntimeError):
    """Cooperative stop requested by the client that owns the question."""


class QuestionDeadlineExceeded(RuntimeError):
    """The bounded execution time for a question has elapsed."""


class PassageDecisionKind(StrEnum):
    """One truthful retrieval decision, including conservative fail-open cases."""

    SECTION_KEPT = "section_kept"
    SECTION_PRUNED = "section_pruned"
    SECTION_CHECK_FAILED = "section_check_failed"
    PASSAGE_RETRIEVED = "passage_retrieved"
    PASSAGE_REJECTED = "passage_rejected"
    PASSAGE_CHECK_FAILED = "passage_check_failed"
    PASSAGE_PRUNED = "passage_pruned"

    @property
    def completes_leaf(self) -> bool:
        return self in (
            self.PASSAGE_RETRIEVED,
            self.PASSAGE_REJECTED,
            self.PASSAGE_CHECK_FAILED,
            self.PASSAGE_PRUNED,
        )

    @property
    def live_group(self) -> str:
        return {
            self.SECTION_KEPT: "kept",
            self.SECTION_PRUNED: "pruned",
            self.SECTION_CHECK_FAILED: "kept",
            self.PASSAGE_RETRIEVED: "retrieved",
            self.PASSAGE_REJECTED: "rejected",
            self.PASSAGE_CHECK_FAILED: "errored",
            self.PASSAGE_PRUNED: "pruned",
        }[self]

    @property
    def audit_status(self) -> str:
        if self in (self.SECTION_CHECK_FAILED, self.PASSAGE_CHECK_FAILED):
            return "error"
        return self.live_group

    @property
    def relevant(self) -> bool:
        return self in (
            self.SECTION_KEPT,
            self.SECTION_CHECK_FAILED,
            self.PASSAGE_RETRIEVED,
        )


@dataclass(frozen=True)
class PassageDecision:
    node_id: str
    kind: PassageDecisionKind
    reason: str = ""
    quote: str = ""
    code: str | None = None
    document_id: str = ""

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("passage decision requires a node identity")
        if self.quote and self.kind is not PassageDecisionKind.PASSAGE_RETRIEVED:
            raise ValueError("only retrieved passages may carry a quote")

    @property
    def identity(self) -> tuple[str, str]:
        return (self.document_id, self.node_id)


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
        self._decisions: dict[tuple[str, str], PassageDecision] = {}
        self._cancelled = threading.Event()
        self._deadline: float | None = None

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def begin_retrieval(self, detail: str = "") -> None:
        self.checkpoint()
        self._set_phase("retrieval", detail)

    def queue(self, detail: str = "") -> None:
        self._set_phase("queued", detail)

    def begin_execution(self, deadline_seconds: float) -> None:
        if deadline_seconds <= 0:
            raise ValueError("question deadline must be positive")
        with self._lock:
            self._deadline = time.monotonic() + deadline_seconds
        self.checkpoint()
        self._set_phase("retrieval", "")

    def set_total(self, total_leaves: int) -> None:
        if total_leaves < 0:
            raise ValueError("retrieval total cannot be negative")
        with self._lock:
            self._total = total_leaves
            self._done = 0
            self._decisions.clear()

    def record_decision(self, decision: PassageDecision) -> None:
        """Record one passage decision and its progress as one locked change."""
        self.checkpoint()
        with self._lock:
            if decision.identity in self._decisions:
                raise RuntimeError(
                    f"retrieval decided passage '{decision.node_id}' twice"
                )
            if decision.kind.completes_leaf:
                if self._done >= self._total:
                    raise RuntimeError("retrieval completed more leaves than declared")
            self._decisions[decision.identity] = decision
            if decision.kind.completes_leaf:
                self._done += 1

    def begin_synthesis(self, detail: str = "") -> None:
        self.checkpoint()
        self._set_phase("synthesis", detail)

    def complete(self, detail: str = "") -> None:
        self._set_phase("completed", detail)

    def fail(self, detail: str = "") -> None:
        self._set_phase("failed", detail)

    def cancel(self, detail: str = "") -> None:
        self._cancelled.set()
        self._set_phase("cancelled", detail)

    def time_out(self, detail: str = "") -> None:
        self._set_phase("timed_out", detail)

    def checkpoint(self) -> None:
        """Stop between model calls without misreporting an unchecked passage."""
        if self._cancelled.is_set():
            raise QuestionCancelled("question was cancelled")
        with self._lock:
            deadline = self._deadline
        if deadline is not None and time.monotonic() >= deadline:
            raise QuestionDeadlineExceeded("question deadline exceeded")

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

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
        groups = {
            "pruned": [],
            "retrieved": [],
            "kept": [],
            "rejected": [],
            "errored": [],
        }
        metadata: dict[str, dict] = {}
        audit: list[dict] = []
        for decision in self._decisions.values():
            key = (
                f"{decision.document_id}::{decision.node_id}"
                if decision.document_id
                else decision.node_id
            )
            groups[decision.kind.live_group].append(key)
            item = {
                "document_id": decision.document_id,
                "node_id": decision.node_id,
                "kind": decision.kind.value,
                "status": decision.kind.audit_status,
                "group": decision.kind.live_group,
                "reason": decision.reason,
                "quote": decision.quote,
                "code": decision.code,
            }
            metadata[key] = {
                "status": item["status"],
                "reason": decision.reason,
                "quote": decision.quote,
            }
            if decision.code is not None:
                metadata[key]["code"] = decision.code
            audit.append(item)
        return {
            **groups,
            "meta": metadata,
            "decisions": audit,
        }

    def _set_phase(self, phase: str, detail: str) -> None:
        with self._lock:
            self._phase = phase
            self._detail = detail
