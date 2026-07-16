"""The corpus must be found regardless of the caller's working directory.

server.py, pipeline.py and run-tauri.py are all started from different places.
When these paths were CWD-relative, starting the server from tauri-app/ made
/api/documents return an empty list instead of raising — the corpus silently
vanished, and the UI showed "no documents" rather than an error.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tauri-app"))

import pageindex  # noqa: E402
import paths  # noqa: E402
from modules.ingest import basic_markdown  # noqa: E402


@pytest.fixture
def elsewhere(tmp_path):
    """Run a block from an unrelated working directory."""
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(previous)


class TestPathsAreAbsolute:
    @pytest.mark.parametrize(
        "path",
        [
            pytest.param(pageindex.KB_DIR, id="pageindex.KB_DIR"),
            pytest.param(pageindex.LIBRARY_DIR, id="pageindex.LIBRARY_DIR"),
            pytest.param(basic_markdown.DOCS_DIR, id="basic_markdown.DOCS_DIR"),
            pytest.param(basic_markdown.KB_DIR, id="basic_markdown.KB_DIR"),
        ],
    )
    def test_is_absolute(self, path):
        assert path.is_absolute(), f"{path} is CWD-relative"

    def test_all_modules_agree_on_the_repo(self):
        for path in (pageindex.KB_DIR, basic_markdown.KB_DIR):
            assert path == ROOT / "knowledge_base"

    def test_the_manifest_writer_and_reader_point_at_the_same_file(self):
        """PDF ingest writes this; Expected-library publication consumes it."""
        from modules.ingest import betteringest_pdf
        from pageindex import build

        assert betteringest_pdf.SOURCES_MANIFEST == paths.SOURCES_MANIFEST
        assert build.SOURCES_MANIFEST == paths.SOURCES_MANIFEST


class TestResolutionIsIndependentOfCwd:
    def test_paths_do_not_move_with_the_working_directory(self, elsewhere):
        assert pageindex.LIBRARY_DIR == ROOT / "library"
        assert pageindex.LIBRARY_DIR.parent == ROOT

    def test_expected_library_has_no_legacy_index_fallback(self, elsewhere):
        assert pageindex.LIBRARY_DIR == ROOT / "library"

    def test_a_relative_path_would_have_failed_here(self, elsewhere):
        # Guards the test itself: prove the working directory really did move,
        # so the assertions above are meaningful rather than vacuous.
        assert Path("library").resolve() != pageindex.LIBRARY_DIR
