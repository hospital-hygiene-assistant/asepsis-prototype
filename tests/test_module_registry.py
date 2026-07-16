"""The ingest adapter registry is the authority for loadable modules."""

import pytest

from modules import registry


def test_load_accepts_a_discovered_ingest_adapter():
    module = registry.load("ingest", "basic_markdown")

    assert module.MODULE_INFO["name"] == "basic_markdown"


@pytest.mark.parametrize(
    ("stage", "name"),
    [
        ("index", "anything"),
        ("ingest", "_captioning"),
        ("ingest", "does_not_exist"),
        ("ingest", "../paths"),
    ],
)
def test_load_rejects_everything_not_exposed_by_discovery(stage, name):
    with pytest.raises(ValueError, match="unknown ingest adapter"):
        registry.load(stage, name)
