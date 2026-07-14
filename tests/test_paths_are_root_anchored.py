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
            pytest.param(pageindex.INDEX_DIR, id="pageindex.INDEX_DIR"),
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
        """PDF ingest writes this; the API reads it to render a source page.

        They each held their own copy of the path, so moving either would have
        broken visual citations without a single test noticing.
        """
        from modules.ingest import betteringest_pdf
        from api import sources

        assert betteringest_pdf.SOURCES_MANIFEST == paths.SOURCES_MANIFEST
        assert sources.SOURCES_MANIFEST == paths.SOURCES_MANIFEST


class TestResolutionIsIndependentOfCwd:
    def test_paths_do_not_move_with_the_working_directory(self, elsewhere):
        assert pageindex.INDEX_DIR == ROOT / "index"
        assert pageindex.INDEX_DIR.parent == ROOT

    def test_index_discovery_works_from_another_directory(self, elsewhere):
        """The concrete regression: globbing the index from the wrong CWD."""
        from_elsewhere = sorted(p.name for p in pageindex.INDEX_DIR.glob("*.json"))
        os.chdir(ROOT)
        from_root = sorted(p.name for p in pageindex.INDEX_DIR.glob("*.json"))
        assert from_elsewhere == from_root

    def test_a_relative_path_would_have_failed_here(self, elsewhere):
        # Guards the test itself: prove the working directory really did move,
        # so the assertions above are meaningful rather than vacuous.
        assert Path("index").resolve() != pageindex.INDEX_DIR
