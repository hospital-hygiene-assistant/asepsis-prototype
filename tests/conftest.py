"""
Shared test fixtures and the verbose per-phase terminal report.

Every logic test in this suite is offline: `fake_ollama` replaces the Ollama
client with a recording stub, so no phase's correctness depends on what a real
model happens to say. Several phases exist specifically to make *fewer* LLM
calls, so the fake records every call and tests assert counts directly.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as app_config  # noqa: E402
import tokens as app_tokens  # noqa: E402


# ---------------------------------------------------------------------------
# Phase labels — drive the grouped summary printed after the run
# ---------------------------------------------------------------------------

PHASES: list[tuple[str, str]] = [
    ("test_config_context|test_tokens|test_runcontext|test_errors", "Phase 0 · foundations"),
    ("test_summaries|test_index_migration",                        "Phase 1 · summaries"),
    ("test_batched_prune",                                         "Phase 2 · batched prune"),
    ("test_ranking|test_budget_eval",                              "Phase 3 · budget + BM25"),
    ("test_manifest_tags|test_betteringest_ingest",                "Phase 4 · tags + ingest"),
    ("test_mcq_precompute",                                        "Phase 5 · choice precompute"),
    ("test_debug_cache",                                           "Phase 6 · debug cache"),
    ("test_pageindex_units|test_chat_endpoint_units|test_retrieval", "Legacy · existing suite"),
]


def _phase_of(nodeid: str) -> str:
    fname = nodeid.split("::")[0].rsplit("/", 1)[-1]
    for pattern, label in PHASES:
        if re.match(rf"^({pattern})\.py$", fname):
            return label
    return "Unclassified"


# ---------------------------------------------------------------------------
# The recording fake Ollama client
# ---------------------------------------------------------------------------

class FakeResponse(dict):
    pass


class FakeOllamaClient:
    """Stands in for ollama.Client.

    Responses come from `handler(prompt, model) -> str | dict`, or from a
    scripted queue, or from the default (a JSON blob that parses cleanly in
    every prompt shape the project uses).
    """

    def __init__(self, recorder: "CallRecorder", handler=None):
        self._recorder = recorder
        self._handler = handler

    def chat(self, model=None, messages=None, options=None, **kw):
        prompt = messages[0]["content"] if messages else ""
        self._recorder.record(prompt=prompt, model=model, options=options or {})
        content = self._recorder.respond(prompt, model)
        return FakeResponse({
            "message": {"content": content},
            # Real Ollama reports this; the token estimator calibrates on it.
            "prompt_eval_count": max(1, len(prompt) // 4),
        })

    # Some call sites use these; keep the surface honest.
    def list(self):
        return {"models": [{"name": m} for m in app_config.MODEL_SPECS]}


class CallRecorder:
    def __init__(self):
        self.calls: list[dict] = []
        self.handler = None
        self.default_response = '{"relevant": false}'
        self.scripted: list[str] = []

    def record(self, **kw):
        self.calls.append(kw)

    def respond(self, prompt: str, model: str | None) -> str:
        if self.handler is not None:
            out = self.handler(prompt, model)
            if out is not None:
                return out if isinstance(out, str) else json.dumps(out)
        if self.scripted:
            return self.scripted.pop(0)
        return self.default_response

    # -- assertions / introspection helpers ---------------------------------

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def prompts(self) -> list[str]:
        return [c["prompt"] for c in self.calls]

    def prompts_containing(self, needle: str) -> list[str]:
        return [p for p in self.prompts if needle in p]

    def reset(self):
        self.calls.clear()

    def assert_count(self, expected: int, what: str = "LLM calls"):
        assert self.count == expected, (
            f"expected {expected} {what}, got {self.count}\n"
            + "\n".join(f"  [{i}] {p[:120]!r}" for i, p in enumerate(self.prompts))
        )

    def assert_every_call_has_num_ctx(self):
        missing = [i for i, c in enumerate(self.calls) if "num_ctx" not in (c["options"] or {})]
        assert not missing, f"calls without num_ctx: {missing}"


@pytest.fixture
def recorder() -> CallRecorder:
    return CallRecorder()


@pytest.fixture
def fake_ollama(monkeypatch, recorder):
    """Patch every Ollama client construction point to the recording fake.

    The teardown matters. `reconfigure_clients` stores client INSTANCES in a
    module global, so monkeypatch reverting `ollama.Client` does not undo it —
    the fake clients stay in the pool for the rest of the session, and every
    later test unknowingly runs against stubs. That silently turned the entire
    live-Ollama tier into a no-op that "failed" in 2.7 seconds.
    """
    import pageindex

    def _factory(*a, **kw):
        return FakeOllamaClient(recorder)

    monkeypatch.setattr(pageindex.ollama, "Client", _factory)
    monkeypatch.setattr(pageindex, "make_client", lambda url: FakeOllamaClient(recorder))
    pageindex.reconfigure_clients(["http://fake:1"])
    yield recorder
    # Rebuild the pool with the REAL client class now restored.
    monkeypatch.undo()
    pageindex.reconfigure_clients(list(_REAL_OLLAMA_URLS))


# The client pool as it was before any test touched it.
_REAL_OLLAMA_URLS: list[str] = []


def pytest_configure(config):
    import pageindex
    _REAL_OLLAMA_URLS[:] = list(pageindex.OLLAMA_URLS)


def real_ollama_available() -> bool:
    """True when the pool holds genuine clients pointing at a live server.

    The live tier must refuse to run against leaked fakes: a stubbed run
    reports confident failures in milliseconds and looks exactly like a real
    retrieval regression.
    """
    import pageindex
    if not pageindex._clients:
        return False
    if any(isinstance(c, FakeOllamaClient) for c in pageindex._clients):
        return False
    try:
        pageindex._clients[0].list()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """Every test starts from default config and a clean token estimator,
    and can never write the developer's real config/runtime.json."""
    monkeypatch.setattr(app_config, "RUNTIME_CONFIG_PATH", tmp_path / "runtime.json")
    app_config.reset_for_tests()
    app_tokens.reset_for_tests()
    yield
    app_config.reset_for_tests()


# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOC = """\
# Guideline
Opening prose that belongs to the root section.

## Diagnosis
How the condition is identified.

### Initial workup
Order a CBC and a metabolic panel before imaging.

### Imaging
CT is preferred over ultrasound for this indication.

## Treatment
Overview of therapy options.

### First-line therapy
Start amoxicillin 500mg three times daily for seven days.

### Second-line therapy
If penicillin-allergic, use doxycycline instead.
"""


@pytest.fixture
def tmp_corpus(tmp_path, monkeypatch):
    """A knowledge_base + index pair rooted in tmp_path, wired into pageindex."""
    import pageindex

    kb = tmp_path / "knowledge_base"
    idx = tmp_path / "index"
    kb.mkdir()
    idx.mkdir()
    (kb / "guideline.md").write_text(SAMPLE_DOC, encoding="utf-8")

    monkeypatch.setattr(pageindex, "KB_DIR", kb)
    monkeypatch.setattr(pageindex, "INDEX_DIR", idx)
    return {"root": tmp_path, "kb": kb, "index": idx, "doc": "guideline"}


# ---------------------------------------------------------------------------
# Grouped terminal summary
# ---------------------------------------------------------------------------

_results: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
_failures: list[tuple[str, str]] = []


def pytest_runtest_logreport(report):
    if report.when != "call" and not (report.when == "setup" and report.outcome == "skipped"):
        return
    phase = _phase_of(report.nodeid)
    _results[phase][report.outcome] += 1
    if report.outcome == "failed":
        _failures.append((phase, report.nodeid))


def pytest_terminal_summary(terminalreporter):
    if not _results:
        return
    w = terminalreporter
    w.write_sep("─", "ASEPSIS RETRIEVAL SUITE", bold=True)

    ordered = [label for _, label in PHASES] + ["Unclassified"]
    width = max((len(l) for l in ordered if l in _results), default=20)

    for label in ordered:
        if label not in _results:
            continue
        counts = _results[label]
        parts = []
        if counts.get("passed"):
            parts.append(("green", f"{counts['passed']:>3} passed"))
        if counts.get("failed"):
            parts.append(("red", f"{counts['failed']:>3} failed"))
        if counts.get("skipped"):
            parts.append(("yellow", f"{counts['skipped']:>3} skipped"))
        w.write(f"  {label.ljust(width)}  ")
        for i, (colour, text) in enumerate(parts):
            if i:
                w.write("  ")
            w.write(text, **{colour: True})
        w.write("\n")

    if _failures:
        w.write("\n")
        w.write_sep("─", "FAILURES", red=True, bold=True)
        for phase, nodeid in _failures:
            w.write(f"  {phase}  ", bold=True)
            w.write(f"{nodeid}\n", red=True)
    else:
        w.write("\n  all selected tests passed\n", green=True, bold=True)
