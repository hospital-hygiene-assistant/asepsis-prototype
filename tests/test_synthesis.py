"""
The shared synthesis module: one prompt for the app and the evaluation, and
the two independent signals read out of an answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tauri-app"))

import config as app_config  # noqa: E402
import synthesis  # noqa: E402

pytestmark = pytest.mark.unit


class TestPrompt:
    def test_completeness_labels_are_included_by_default(self):
        app_config.update(completeness_check=True)
        p = synthesis.build_synthesis_prompt("q", "[1] x")
        assert "EVIDENCE_SUFFICIENT" in p and "STILL_NEEDED" in p

    def test_completeness_labels_follow_the_runtime_setting(self):
        app_config.update(completeness_check=False)
        assert "EVIDENCE_SUFFICIENT" not in synthesis.build_synthesis_prompt("q", "p")
        # An explicit argument overrides the setting either way.
        assert "EVIDENCE_SUFFICIENT" in synthesis.build_synthesis_prompt("q", "p", completeness=True)

    def test_braces_in_query_or_passages_survive(self):
        p = synthesis.build_synthesis_prompt("dose {mg}", "[1] {json: true}")
        assert "dose {mg}" in p and "{json: true}" in p

    def test_server_shows_the_prompt_it_sends(self):
        """/api/chat/config displays the template; it must be the same text
        the handler builds, with only the placeholders left in."""
        import server
        shown = server._synthesis_prompt_template()
        real = synthesis.build_synthesis_prompt("QQQ", "PPP", completeness=True)
        assert shown.replace("{query}", "QQQ").replace("{passages}", "PPP") == real

    def test_passage_format_matches_the_handler(self):
        s = {"n": 2, "doc": "icu_sedation_guide", "breadcrumb": "A › B", "title": "B",
             "excerpt": "text"}
        assert synthesis.format_passage(s) == "[2] icu sedation guide › A › B\ntext"
        s["breadcrumb"] = ""
        assert synthesis.format_passage(s).startswith("[2] icu sedation guide › B\n")


class TestGrounding:
    def test_three_statuses(self):
        assert synthesis.grounding_status("", 0)[0] == "insufficient_evidence"
        assert synthesis.grounding_status("no cites", 3)[0] == "partially_grounded"
        assert synthesis.grounding_status("yes [2]", 3) == ("grounded", {2})

    def test_out_of_range_citation_is_not_grounding(self):
        assert synthesis.grounding_status("see [7]", 3)[0] == "partially_grounded"


class TestJudgment:
    def _j(self, text):
        return synthesis.interpret_judgment(synthesis.parse_answer_sections(text))

    def test_plain_yes_and_no(self):
        assert self._j("SHORT_ANSWER: x\nEVIDENCE_SUFFICIENT: yes")["label"] == "sufficient"
        j = self._j("SHORT_ANSWER: x\nEVIDENCE_SUFFICIENT: No.\nSTILL_NEEDED: - the dose\n- the age")
        assert j["label"] == "insufficient"
        assert j["still_needed"] == ["the dose", "the age"]

    def test_missing_label_is_reported_not_defaulted(self):
        j = self._j("SHORT_ANSWER: only this")
        assert j["raw"] is None
        assert j["label"] == "missing"
        # ...and the product's prefix match would read that as sufficient.
        assert j["ui_reading"] == "sufficient"

    def test_hedged_label_is_unparsed_and_ui_reads_it_as_sufficient(self):
        j = self._j("SHORT_ANSWER: x\nEVIDENCE_SUFFICIENT: Partially — the dose is covered")
        assert j["label"] == "unparsed"
        assert j["raw"].startswith("Partially")
        assert j["ui_reading"] == "sufficient"

    def test_ui_and_strict_agree_on_the_easy_cases(self):
        for text, expected in [("no", "insufficient"), ("Yes", "sufficient"),
                               ("NO — missing the age", "insufficient")]:
            j = self._j(f"EVIDENCE_SUFFICIENT: {text}")
            assert j["label"] == expected and j["ui_reading"] == expected

    def test_bold_markdown_label_still_parses(self):
        j = self._j("**EVIDENCE_SUFFICIENT:** **no**\n**STILL_NEEDED:** - x")
        assert j["label"] == "insufficient"


class TestSynthesiseCall:
    def test_streams_and_parses(self, monkeypatch):
        app_config.update(completeness_check=True)
        seen = {}

        class FakeClient:
            def chat(self, **kw):
                seen.update(kw)
                for piece in ["SHORT_ANSWER: A [1].\n", "EVIDENCE_SUFFICIENT: no\n",
                              "STILL_NEEDED: - weight"]:
                    yield {"message": {"content": piece}, "prompt_eval_count": 0}
                yield {"message": {"content": ""}, "prompt_eval_count": 123}

        sources = synthesis.number_sources([{"doc": "d", "node_id": "n", "title": "t",
                                             "breadcrumb": "", "excerpt": "e"}])
        res = synthesis.synthesise("q?", sources, client=FakeClient(), model="m")
        assert seen["stream"] is True and seen["model"] == "m"
        assert "num_ctx" in seen["options"]
        assert res.sections["short_answer"] == "A [1]."
        assert res.judgment["label"] == "insufficient"
        assert res.judgment["still_needed"] == ["weight"]
        assert res.prompt_eval_count == 123
        assert res.ms >= 0

    def test_cancel_stops_the_stream(self):
        class FakeClient:
            def chat(self, **kw):
                yield {"message": {"content": "x"}}
                yield {"message": {"content": "y"}}

        with pytest.raises(synthesis.SynthesisCancelled):
            synthesis.synthesise("q", [], client=FakeClient(), model="m",
                                 should_cancel=lambda: True)
