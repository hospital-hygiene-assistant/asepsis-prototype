"""
Static checks over the project's own source.

These exist because of a specific, repeated failure: code paths that no test
exercises — the PDF ingest orchestration especially — accumulated references to
names that did not exist. `_manifest` and `bpdf` both reached the user as a
runtime NameError on a button click, from two different functions, and nothing
in a 297-test suite noticed, because nothing calls those functions.

Undefined names are exactly what a linter catches for free. Behavioural tests
are still the right tool for behaviour; this is the floor beneath them.
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent

# Our own code. The vendored BetterIngester tree under _betteringest/ is
# excluded — it is third-party, and we do not want its style noise here.
SOURCES = [
    "pageindex.py", "config.py", "tokens.py", "ranking.py", "choices.py",
    "debug_cache.py", "query.py", "pipeline.py", "ingest.py",
    "run_tests.py", "install_dependencies.py",
    "modules/registry.py",
    "modules/ingest/_manifest.py",
    "modules/ingest/betteringest_pdf.py",
    "modules/ingest/basic_markdown.py",
    "modules/ingest/_massage.py",
    "modules/ingest/_captioning.py",
    "modules/index/pageindex_custom.py",
    "modules/query/ollama_synthesis.py",
    "tauri-app/server.py",
]

# Failures that mean "this will raise at runtime". Unused imports and the like
# are not included: this check is a correctness floor, not a style gate, and a
# noisy gate is one people learn to ignore.
FATAL = ("undefined name", "syntax error")


def _existing_sources() -> list[str]:
    return [s for s in SOURCES if (ROOT / s).exists()]


@pytest.fixture(scope="module")
def pyflakes_output() -> str:
    try:
        import pyflakes  # noqa: F401
    except ImportError:
        pytest.skip("pyflakes not installed — see requirements-dev.txt")
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", *_existing_sources()],
        cwd=ROOT, capture_output=True, text=True)
    return proc.stdout + proc.stderr


def test_no_undefined_names(pyflakes_output):
    """The regression guard. Two of these shipped to the user as NameErrors on
    the ingest confirm button, in functions no test covers."""
    bad = [line for line in pyflakes_output.splitlines()
           if any(f in line.lower() for f in FATAL)]
    assert not bad, (
        "pyflakes found names that do not exist at runtime:\n  "
        + "\n  ".join(bad))


def test_every_source_file_is_checked():
    """A file dropped from SOURCES would silently stop being checked."""
    missing = [s for s in SOURCES if not (ROOT / s).exists()]
    assert not missing, f"listed but absent (stale entry?): {missing}"


def test_all_modules_import_cleanly():
    """Import every first-party module in a subprocess.

    Catches import-time errors that pyflakes cannot see — a bad relative
    import, a module-level call that raises — without polluting this process.
    """
    mods = ["config", "tokens", "ranking", "choices", "debug_cache",
            "pageindex", "query",
            "modules.registry", "modules.ingest._manifest",
            "modules.ingest.basic_markdown", "modules.ingest.betteringest_pdf",
            "modules.index.pageindex_custom", "modules.query.ollama_synthesis"]
    code = "import sys; sys.path.insert(0, '.');" + "".join(
        f"__import__({m!r});" for m in mods)
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"import failed:\n{proc.stderr[-1500:]}"
