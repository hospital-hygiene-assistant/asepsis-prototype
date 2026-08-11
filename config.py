"""
Central model + context configuration.

Every LLM call in this project resolves its context window from here — no call
site carries a hardcoded number. Before this module existed, no call passed
`num_ctx` at all, so Ollama silently applied its 4096-token default and
truncated long synthesis prompts without an error.

Two numbers matter per model:

  max_ctx      the model's ceiling (gemma4:e4b: 128k)
  default_ctx  what we actually run at — 48k, inside the 32-64k band where
               quality holds up for multi-passage reasoning

The live values sit in RuntimeConfig, are persisted to config/runtime.json,
and are adjustable from the frontend at runtime (see /api/config). Raising
`agent_ctx` widens the Phase-3 retrieval budget, so the deferred-node count
moves with the setting.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME_CONFIG_PATH = ROOT / "config" / "runtime.json"

# Bumped whenever the on-disk index format changes in a way that requires a
# rebuild (see pageindex.build_index / the migration check in the server).
INDEX_FORMAT_VERSION = 2

# Bumped per-prompt whenever wording changes, so summaries, facet precomputes
# and debug-cache entries all invalidate correctly.
PROMPT_VERSIONS: dict[str, int] = {
    "leaf_summary": 1,
    "section_summary": 1,
    "child_select": 1,
    "leaf_eval": 1,
    "explain": 1,
    "facet_leaf": 1,
    "synthesis": 1,
}


@dataclass(frozen=True)
class ModelSpec:
    """Static capabilities of a model tag."""
    name: str
    max_ctx: int          # hard ceiling supported by the model
    default_ctx: int      # working default — the quality band, not the ceiling
    reserve: int          # tokens held back for the model's own response

    def clamp(self, ctx: int) -> int:
        return max(MIN_CTX, min(int(ctx), self.max_ctx))


MIN_CTX = 4096

MODEL_SPECS: dict[str, ModelSpec] = {
    "gemma4:e4b": ModelSpec(
        name="gemma4:e4b",
        max_ctx=131072,      # 128k
        default_ctx=49152,   # 48k — mid 32-64k band
        reserve=2048,
    ),
}

DEFAULT_RETRIEVAL_MODEL = "gemma4:e4b"
DEFAULT_SYNTHESIS_MODEL = "gemma4:e4b"

# Applied to any tag we have no entry for. Deliberately modest, and always
# accompanied by a warning — an unknown model must never silently inherit
# Ollama's 4096 default again, nor pretend to a window it may not have.
FALLBACK_SPEC = ModelSpec(name="(unknown)", max_ctx=32768,
                          default_ctx=8192, reserve=1024)

_warned_unknown: set[str] = set()


def spec_for(model: str) -> ModelSpec:
    """Resolve a model tag to its spec. Unknown tags warn once, then fall back."""
    known = MODEL_SPECS.get(model)
    if known is not None:
        return known
    if model not in _warned_unknown:
        _warned_unknown.add(model)
        import sys
        print(
            f"[config] WARNING: no ModelSpec for '{model}'. Falling back to "
            f"num_ctx={FALLBACK_SPEC.default_ctx} (max {FALLBACK_SPEC.max_ctx}). "
            f"Add it to MODEL_SPECS in config.py to use its real context window.",
            file=sys.stderr,
        )
    return ModelSpec(name=model, max_ctx=FALLBACK_SPEC.max_ctx,
                     default_ctx=FALLBACK_SPEC.default_ctx,
                     reserve=FALLBACK_SPEC.reserve)


@dataclass
class RuntimeConfig:
    """The live, user-adjustable settings.

    `agent_ctx` is the important one: it is the window the *answering* model
    reasons in, and therefore the budget that decides how many retrieved nodes
    survive to the answer versus being deferred.
    """
    retrieval_model: str = DEFAULT_RETRIEVAL_MODEL
    synthesis_model: str = DEFAULT_SYNTHESIS_MODEL
    retrieval_ctx: int = 0     # 0 → use the spec default
    agent_ctx: int = 0         # 0 → use the spec default
    concurrency_per_instance: int = 2
    max_leaf_evals: int = 400  # wall-clock guard, independent of the token budget
    debug_cache_enabled: bool = False

    def resolved_retrieval_ctx(self) -> int:
        spec = spec_for(self.retrieval_model)
        return spec.clamp(self.retrieval_ctx or spec.default_ctx)

    def resolved_agent_ctx(self) -> int:
        spec = spec_for(self.synthesis_model)
        return spec.clamp(self.agent_ctx or spec.default_ctx)


_lock = threading.Lock()
_runtime: RuntimeConfig | None = None


def runtime() -> RuntimeConfig:
    """The process-wide runtime config, loaded from disk on first access."""
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = _load()
        return _runtime


def update(**kw) -> RuntimeConfig:
    """Apply and persist a partial update. Unknown keys are ignored."""
    global _runtime
    with _lock:
        cfg = _runtime if _runtime is not None else _load()
        for key, value in kw.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        # Clamp the context settings against the (possibly just-changed) models.
        if cfg.retrieval_ctx:
            cfg.retrieval_ctx = spec_for(cfg.retrieval_model).clamp(cfg.retrieval_ctx)
        if cfg.agent_ctx:
            cfg.agent_ctx = spec_for(cfg.synthesis_model).clamp(cfg.agent_ctx)
        _runtime = cfg
        _save(cfg)
        return cfg


def reset_for_tests(cfg: RuntimeConfig | None = None) -> None:
    """Test hook: install a config without touching disk."""
    global _runtime
    with _lock:
        _runtime = cfg if cfg is not None else RuntimeConfig()


def _load() -> RuntimeConfig:
    try:
        data = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
        fields = {f for f in RuntimeConfig().__dict__}
        return RuntimeConfig(**{k: v for k, v in data.items() if k in fields})
    except Exception:
        return RuntimeConfig()


def _save(cfg: RuntimeConfig) -> None:
    try:
        RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        RUNTIME_CONFIG_PATH.write_text(
            json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    except OSError as exc:
        import sys
        print(f"[config] could not persist runtime config: {exc}", file=sys.stderr)


def chat_options(kind: str = "retrieval", *, temperature: float = 0) -> dict:
    """The `options` dict for an ollama chat call. `kind` is 'retrieval' or 'agent'."""
    cfg = runtime()
    ctx = cfg.resolved_agent_ctx() if kind == "agent" else cfg.resolved_retrieval_ctx()
    return {"temperature": temperature, "num_ctx": ctx}


def describe() -> dict:
    """Serialisable view of the current configuration, for /api/config."""
    cfg = runtime()
    r_spec, s_spec = spec_for(cfg.retrieval_model), spec_for(cfg.synthesis_model)
    return {
        "retrieval_model": cfg.retrieval_model,
        "synthesis_model": cfg.synthesis_model,
        "retrieval_ctx": cfg.resolved_retrieval_ctx(),
        "agent_ctx": cfg.resolved_agent_ctx(),
        "retrieval_ctx_max": r_spec.max_ctx,
        "agent_ctx_max": s_spec.max_ctx,
        "agent_ctx_reserve": s_spec.reserve,
        "recommended_ctx_band": [32768, 65536],
        "min_ctx": MIN_CTX,
        "concurrency_per_instance": cfg.concurrency_per_instance,
        "max_leaf_evals": cfg.max_leaf_evals,
        "debug_cache_enabled": cfg.debug_cache_enabled,
        "known_models": sorted(MODEL_SPECS),
        "index_format_version": INDEX_FORMAT_VERSION,
    }
