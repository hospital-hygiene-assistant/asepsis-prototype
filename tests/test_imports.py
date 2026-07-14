"""Every module must import.

A broken import is invisible to both the suite and to pyflakes: pyflakes checks
names within a file and never resolves `from .x import y`, and a module nothing
imports is a module nothing tests. That is how a module once shipped importing a
renamed sibling with the whole suite green.
"""

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tauri-app"))


def _modules_under(package_name: str) -> list[str]:
    package = importlib.import_module(package_name)
    return sorted(
        name
        for _, name, _ in pkgutil.walk_packages(package.__path__, f"{package_name}.")
    )


@pytest.mark.parametrize("name", _modules_under("pageindex"))
def test_pageindex_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", _modules_under("api"))
def test_api_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", _modules_under("modules"))
def test_pipeline_module_imports(name):
    # The OCR path imports paddle lazily, so this stays honest without it.
    importlib.import_module(name)


@pytest.mark.parametrize("name", ["server", "pipeline", "paths", "retrieval_cases"])
def test_top_level_module_imports(name):
    importlib.import_module(name)
