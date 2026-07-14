"""Live state of the current chat run.

Process-wide: one run at a time, which holds for a local single-user session.
A shared deployment needs this keyed per run before two clients can be served
at once, or one client's phase will overwrite another's.
"""

import threading

_lock = threading.Lock()
_phase: dict = {"phase": "idle", "detail": ""}


def set_chat_phase(phase: str, detail: str = "") -> None:
    with _lock:
        _phase.update({"phase": phase, "detail": detail})


def chat_phase() -> dict:
    with _lock:
        return dict(_phase)


def is_error() -> bool:
    with _lock:
        return _phase["phase"] == "error"
