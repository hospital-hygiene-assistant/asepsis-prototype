"""The Ollama client pool.

Set OLLAMA_URLS=url1,url2,... to run several instances: each is an independent
process, and branches are handed out round-robin so N instances work on N
branches at once.
"""

import os
import threading
from collections.abc import Callable

import ollama

OLLAMA_URLS: list[str] = [
    u.strip()
    for u in os.getenv("OLLAMA_URLS", "http://localhost:11434").split(",")
    if u.strip()
]

REQUEST_TIMEOUT_SECONDS = float(os.getenv("ASEPSIS_OLLAMA_REQUEST_TIMEOUT", "180"))


def _client(url: str) -> ollama.Client:
    return ollama.Client(host=url, timeout=REQUEST_TIMEOUT_SECONDS)


_clients: list[ollama.Client] = [_client(url) for url in OLLAMA_URLS]
_rr_lock = threading.Lock()
_rr_index = 0

# {url: leaves in flight}, so a client can show which instance is busy.
_activity: dict[str, int] = {url: 0 for url in OLLAMA_URLS}
_activity_lock = threading.Lock()
_slots: dict[str, threading.Semaphore] = {
    url: threading.Semaphore(1) for url in OLLAMA_URLS
}


def make_client(url: str) -> ollama.Client:
    return _client(url)


def round_robin_client() -> tuple[ollama.Client, str]:
    global _rr_index
    with _rr_lock:
        idx = _rr_index % len(_clients)
        client, url = _clients[idx], OLLAMA_URLS[idx]
        _rr_index += 1
    return client, url


def reconfigure_clients(urls: list[str]) -> None:
    """Swap the pool. Called when the instance count changes."""
    global OLLAMA_URLS, _clients, _rr_index
    with _rr_lock:
        OLLAMA_URLS = urls
        _clients = [_client(u) for u in urls]
        _rr_index = 0
    with _activity_lock:
        _activity.clear()
        for url in urls:
            _activity[url] = 0
        _slots.clear()
        for url in urls:
            _slots[url] = threading.Semaphore(1)


def acquire(url: str, checkpoint: Callable[[], None] | None = None) -> None:
    """Claim one real Ollama process, remaining cancellable while queued."""
    with _activity_lock:
        slot = _slots.setdefault(url, threading.Semaphore(1))
    while not slot.acquire(timeout=0.1):
        if checkpoint is not None:
            checkpoint()
    try:
        if checkpoint is not None:
            checkpoint()
    except BaseException:
        slot.release()
        raise
    with _activity_lock:
        _activity[url] = _activity.get(url, 0) + 1


def release(url: str) -> None:
    with _activity_lock:
        _activity[url] = max(0, _activity.get(url, 0) - 1)
        slot = _slots.get(url)
    if slot is not None:
        slot.release()


def get_activity() -> dict[str, int]:
    with _activity_lock:
        return dict(_activity)


def pool() -> list[tuple[ollama.Client, str]]:
    """Every (client, url) in the pool, in port order."""
    with _rr_lock:
        return list(zip(_clients, OLLAMA_URLS))
