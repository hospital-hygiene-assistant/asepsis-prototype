"""One bounded FIFO lifecycle for every local question."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
import traceback
from typing import Any

from pageindex import QuestionCancelled, QuestionDeadlineExceeded, QuestionRun

TTL_SECONDS = 30 * 60
MAX_RUNS = 64
MAX_QUEUED_RUNS = 8
RUN_DEADLINE_SECONDS = 15 * 60
TERMINAL_PHASES = {"completed", "failed", "cancelled", "timed_out"}


class DuplicateRunId(ValueError):
    pass


class RunQueueFull(RuntimeError):
    pass


class RunRegistryFull(RuntimeError):
    pass


@dataclass
class _RunRecord:
    run: QuestionRun
    kind: str
    work: Callable[[QuestionRun], dict[str, Any]]
    future: Future | None = None
    result: dict[str, Any] | None = None
    error_code: str | None = None
    terminal_at: float | None = None


class RunRegistry:
    """Retain results and execute submitted questions one at a time."""

    def __init__(
        self,
        ttl: float = TTL_SECONDS,
        max_runs: int = MAX_RUNS,
        max_queued: int = MAX_QUEUED_RUNS,
        deadline: float = RUN_DEADLINE_SECONDS,
    ) -> None:
        self._records: dict[str, _RunRecord] = {}
        self._lock = threading.Lock()
        self._ttl = ttl
        self._max = max_runs
        self._max_queued = max_queued
        self._deadline = deadline
        self._executor = self._new_executor()
        self._shutdown = False

    def submit(
        self,
        kind: str,
        work: Callable[[QuestionRun], dict[str, Any]],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        run = QuestionRun(run_id)
        run.queue()
        with self._lock:
            if self._shutdown:
                self._executor = self._new_executor()
                self._shutdown = False
            self._evict_unlocked()
            if run.id in self._records:
                raise DuplicateRunId(run.id)
            queued = sum(
                record.run.phase == "queued" for record in self._records.values()
            )
            if queued >= self._max_queued:
                raise RunQueueFull("question queue is full")
            record = _RunRecord(run, kind, work)
            self._records[run.id] = record
            record.future = self._executor.submit(self._execute, record)
            return self._snapshot_unlocked(record)

    def snapshot(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(run_id)
            return self._snapshot_unlocked(record) if record is not None else None

    def cancel(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return None
            if record.run.phase not in TERMINAL_PHASES:
                record.run.cancel()
                if record.future is not None:
                    record.future.cancel()
                record.terminal_at = time.monotonic()
            return self._snapshot_unlocked(record)

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            for record in self._records.values():
                if record.run.phase not in TERMINAL_PHASES:
                    record.run.cancel("backend shutting down")
                    record.terminal_at = time.monotonic()
        self._executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _new_executor() -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="asepsis-question",
        )

    def _execute(self, record: _RunRecord) -> None:
        try:
            record.run.begin_execution(self._deadline)
            result = record.work(record.run)
            record.run.checkpoint()
            with self._lock:
                record.result = result
            record.run.complete()
        except QuestionCancelled:
            record.run.cancel()
        except QuestionDeadlineExceeded:
            record.run.time_out()
            with self._lock:
                record.error_code = "deadline_exceeded"
        except Exception:
            traceback.print_exc()
            record.run.fail("unexpected_failure")
            with self._lock:
                record.error_code = "unexpected_failure"
        finally:
            with self._lock:
                record.terminal_at = time.monotonic()

    def _snapshot_unlocked(self, record: _RunRecord) -> dict[str, Any]:
        state = record.run.phase
        queued = sorted(
            (item for item in self._records.values() if item.run.phase == "queued"),
            key=lambda item: item.run.started,
        )
        position = next(
            (index for index, item in enumerate(queued, 1) if item is record),
            None,
        )
        return {
            **record.run.snapshot(),
            "kind": record.kind,
            "queue_position": position,
            "result": record.result if state == "completed" else None,
            "error": {"code": record.error_code} if record.error_code else None,
        }

    def _evict_unlocked(self) -> None:
        now = time.monotonic()
        for run_id in [
            run_id for run_id, record in self._records.items()
            if record.terminal_at is not None
            and now - record.terminal_at >= self._ttl
        ]:
            del self._records[run_id]
        while len(self._records) >= self._max:
            terminal = [
                record for record in self._records.values()
                if record.terminal_at is not None
            ]
            if not terminal:
                raise RunRegistryFull("all tracked question runs are active")
            oldest = min(terminal, key=lambda record: record.terminal_at or 0)
            del self._records[oldest.run.id]


registry = RunRegistry()
