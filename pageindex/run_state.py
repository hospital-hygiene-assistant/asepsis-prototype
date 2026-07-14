"""Live state of a retrieval run: progress, and each node's verdict.

A run spans one retrieve call per document, so the state outlives any one of
them. Worker threads write it while /api/status reads it, hence the lock.

There is one run per process. Two clients retrieving at once share it, so the
second start_run resets the first's progress and /api/status reports a blend of
both. Serving concurrent clients means keying this by run id and handing the id
to the caller.
"""

import threading

_lock = threading.Lock()

_total = 0
_done = 0

_pruned: set[str] = set()      # sections pruned, and their descendants
_retrieved: set[str] = set()   # leaves the model judged relevant
_kept: set[str] = set()        # sections that passed the prune check
_rejected: set[str] = set()    # leaves evaluated, verdict "not relevant"
_errored: set[str] = set()     # leaves the model could not evaluate at all —
                               # apart from _rejected because "not checked" is
                               # not "irrelevant"

# {node_id: {status, reason, quote}}, written as each verdict is made so a
# client can watch the run rather than wait for it.
_meta: dict[str, dict] = {}


def start_run(total_leaves: int) -> None:
    """Begin a pass over `total_leaves`, discarding the previous run."""
    global _total, _done
    with _lock:
        _total = total_leaves
        _done = 0
        for group in (_pruned, _retrieved, _kept, _rejected, _errored):
            group.clear()
        _meta.clear()


def leaf_done() -> None:
    global _done
    with _lock:
        _done += 1


def get_progress() -> dict[str, int]:
    with _lock:
        return {"total": _total, "done": _done}


def mark_pruned(node_id: str) -> None:
    with _lock:
        _pruned.add(node_id)


def mark_retrieved(node_id: str) -> None:
    with _lock:
        _retrieved.add(node_id)


def mark_kept(node_id: str) -> None:
    with _lock:
        _kept.add(node_id)


def mark_rejected(node_id: str) -> None:
    with _lock:
        _rejected.add(node_id)


def mark_errored(node_id: str) -> None:
    with _lock:
        _errored.add(node_id)


def set_meta(node_id: str, status: str, reason: str = "", quote: str = "") -> None:
    with _lock:
        _meta[node_id] = {"status": status, "reason": reason, "quote": quote}


def get_live_events() -> dict:
    with _lock:
        return {
            "pruned": list(_pruned),
            "retrieved": list(_retrieved),
            "kept": list(_kept),
            "rejected": list(_rejected),
            "errored": list(_errored),
            "meta": {k: dict(v) for k, v in _meta.items()},
        }
