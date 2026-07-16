"""Committed chat fixtures must be exact outputs of the v3 producer."""

import json
from pathlib import Path

from api.chat_contract_fixtures import fixture_payloads, main


FIXTURES = Path(__file__).parent / "contracts" / "chat_v3"


def test_all_outcome_fixtures_match_the_authoritative_producer():
    payloads = fixture_payloads()

    assert set(payloads) == {
        "answered_complete.json",
        "answered_partial.json",
        "insufficient_evidence.json",
        "search_incomplete.json",
        "retrieval_unavailable.json",
        "synthesis_unavailable_complete.json",
        "synthesis_unavailable_partial.json",
    }
    for filename, payload in payloads.items():
        assert (FIXTURES / filename).read_text(encoding="utf-8") == payload


def test_answered_complete_fixture_covers_an_exact_multi_page_visual():
    payload = json.loads(fixture_payloads()["answered_complete.json"])

    visual = payload["outcome"]["citations"][0]["visual"]
    assert visual["status"] == "exact"
    assert [page["page"] for page in visual["pages"]] == [2, 3]


def test_fixture_cli_writes_and_checks_every_target_directory(tmp_path):
    backend = tmp_path / "backend"
    frontend = tmp_path / "frontend"

    assert main(["--write", str(backend), str(frontend)]) == 0
    assert main(["--check", str(backend), str(frontend)]) == 0
    assert {
        path.name for path in backend.iterdir()
    } == set(fixture_payloads())
    assert {
        path.name for path in frontend.iterdir()
    } == set(fixture_payloads())


def test_fixture_cli_check_fails_for_stale_or_extra_contracts(tmp_path):
    target = tmp_path / "fixtures"
    assert main(["--write", str(target)]) == 0
    (target / "answered_complete.json").write_text("stale", encoding="utf-8")
    (target / "obsolete.json").write_text("{}", encoding="utf-8")

    assert main(["--check", str(target)]) == 1
