"""The live state of one retrieval run: progress, and each node's verdict.

A run spans one retrieve call per document, so its state outlives any single
call and is passed in rather than reached for. Worker threads write it while a
client polls it, hence the lock.

One object per run is what keeps concurrent clients apart. Sharing one would
mean the second run resetting the first's progress, and both reading a blend.
"""

import threading


class RunState:
    """One retrieval pass. Safe to write from workers while a client reads it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._done = 0
        self._pruned: set[str] = set()      # sections pruned, and their descendants
        self._retrieved: set[str] = set()   # leaves the model judged relevant
        self._kept: set[str] = set()        # sections that passed the prune check
        self._rejected: set[str] = set()    # leaves evaluated, verdict "not relevant"
        self._errored: set[str] = set()     # leaves the model could not evaluate —
                                            # apart from rejected, because "not
                                            # checked" is not "irrelevant"
        self._meta: dict[str, dict] = {}    # {node_id: {status, reason, quote}}

    def start(self, total_leaves: int) -> None:
        """Begin a pass over `total_leaves`, discarding anything recorded before."""
        with self._lock:
            self._total = total_leaves
            self._done = 0
            for group in (self._pruned, self._retrieved, self._kept,
                          self._rejected, self._errored):
                group.clear()
            self._meta.clear()

    def leaf_done(self) -> None:
        with self._lock:
            self._done += 1

    def progress(self) -> dict[str, int]:
        with self._lock:
            return {"total": self._total, "done": self._done}

    def mark(self, status: str, node_id: str) -> None:
        """Record a node in one of the verdict groups."""
        groups = {
            "pruned": self._pruned,
            "retrieved": self._retrieved,
            "kept": self._kept,
            "rejected": self._rejected,
            "errored": self._errored,
        }
        with self._lock:
            groups[status].add(node_id)

    def set_meta(self, node_id: str, status: str, reason: str = "", quote: str = "") -> None:
        """Record why a node got its verdict, as it is decided."""
        with self._lock:
            self._meta[node_id] = {"status": status, "reason": reason, "quote": quote}

    def events(self) -> dict:
        with self._lock:
            return {
                "pruned": list(self._pruned),
                "retrieved": list(self._retrieved),
                "kept": list(self._kept),
                "rejected": list(self._rejected),
                "errored": list(self._errored),
                "meta": {k: dict(v) for k, v in self._meta.items()},
            }
