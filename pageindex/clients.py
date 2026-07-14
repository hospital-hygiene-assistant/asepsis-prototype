"""The Ollama client pool.

Set OLLAMA_URLS=url1,url2,... to run several instances: each is an independent
process, and branches are handed out round-robin so N instances work on N
branches at once.
"""

import os
import threading

import ollama

OLLAMA_URLS: list[str] = [
    u.strip()
    for u in os.getenv("OLLAMA_URLS", "http://localhost:11434").split(",")
    if u.strip()
]

_clients: list[ollama.Client] = [ollama.Client(host=url) for url in OLLAMA_URLS]
_rr_lock = threading.Lock()
_rr_index = 0

# {url: leaves in flight}, so a client can show which instance is busy.
_activity: dict[str, int] = {url: 0 for url in OLLAMA_URLS}
_activity_lock = threading.Lock()


def make_client(url: str) -> ollama.Client:
    return ollama.Client(host=url)


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
        _clients = [ollama.Client(host=u) for u in urls]
        _rr_index = 0
    with _activity_lock:
        _activity.clear()
        for url in urls:
            _activity[url] = 0


def acquire(url: str) -> None:
    with _activity_lock:
        _activity[url] = _activity.get(url, 0) + 1


def release(url: str) -> None:
    with _activity_lock:
        _activity[url] = max(0, _activity.get(url, 0) - 1)


def get_activity() -> dict[str, int]:
    with _activity_lock:
        return dict(_activity)


def pool() -> list[tuple[ollama.Client, str]]:
    """Every (client, url) in the pool, in port order."""
    with _rr_lock:
        return list(zip(_clients, OLLAMA_URLS))
