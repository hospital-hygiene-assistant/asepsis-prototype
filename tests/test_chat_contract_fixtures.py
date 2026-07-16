"""Committed chat fixtures must be exact outputs of the v2 producer."""

import json
from pathlib import Path

from api.chat_contract_fixtures import fixture_payloads


FIXTURES = Path(__file__).parent / "contracts" / "chat_v2"


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

    visual = payload["grounding"]["sources"][0]["visual"]
    assert visual["status"] == "exact"
    assert [page["page"] for page in visual["pages"]] == [2, 3]
