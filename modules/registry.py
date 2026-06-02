"""
Module registry — discovers available implementations for each pipeline stage
by scanning modules/<stage>/*.py and reading their MODULE_INFO dicts.

Dropping a new .py file (with MODULE_INFO) into modules/ingest/, modules/index/,
or modules/query/ is enough to register it — no other changes needed.
"""
import importlib
import sys
from pathlib import Path
from typing import Any

STAGES = ["ingest", "index", "query"]
_BASE  = Path(__file__).parent


def discover() -> dict[str, dict[str, dict]]:
    """
    Returns:
        {
          "ingest": {"basic_markdown": MODULE_INFO, ...},
          "index":  {"pageindex_custom": MODULE_INFO, ...},
          "query":  {"ollama_synthesis": MODULE_INFO, ...},
        }
    """
    # Ensure the project root is on sys.path so module files can import from it
    root = str(_BASE.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    result: dict[str, dict[str, dict]] = {s: {} for s in STAGES}

    for stage in STAGES:
        stage_dir = _BASE / stage
        if not stage_dir.exists():
            continue
        for py in sorted(stage_dir.glob("*.py")):
            if py.name.startswith("_"):
                continue
            mod_path = f"modules.{stage}.{py.stem}"
            try:
                mod = importlib.import_module(mod_path)
                info = getattr(mod, "MODULE_INFO", None)
                if info and isinstance(info, dict) and "name" in info:
                    result[stage][info["name"]] = info
            except Exception as exc:
                print(f"[registry] warning: could not load {mod_path}: {exc}")

    return result


def load(stage: str, name: str) -> Any:
    """Import and return the module object for the given stage + name."""
    root = str(_BASE.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(f"modules.{stage}.{name}")


def defaults() -> dict[str, str]:
    """Returns the default module name for each stage."""
    return {
        "ingest": "basic_markdown",
        "index":  "pageindex_custom",
        "query":  "ollama_synthesis",
    }
