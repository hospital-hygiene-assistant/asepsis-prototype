"""Deterministic unit tests for the chat endpoint helpers (no Ollama needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tauri-app"))

from server import _parse_answer_sections  # noqa: E402


class TestParseAnswerSections:
    def test_all_four_sections(self):
        text = (
            "SHORT_ANSWER: Limit sodium to under 2,300 mg/day [1].\n"
            "RECOMMENDED_ACTION: Advise a 1,500 mg/day target [1].\n"
            "RATIONALE: Reductions lower systolic BP by 5-6 mmHg [1][2].\n"
            "LIMITATIONS: The sources do not cover pediatric patients."
        )
        s = _parse_answer_sections(text)
        assert s["short_answer"] == "Limit sodium to under 2,300 mg/day [1]."
        assert s["recommended_action"].startswith("Advise a 1,500")
        assert "[1][2]" in s["rationale"]
        assert "pediatric" in s["limitations"]

    def test_markdown_bold_labels(self):
        text = "**SHORT_ANSWER:** Yes [1].\n**RATIONALE:** Because [1]."
        s = _parse_answer_sections(text)
        assert s["short_answer"] == "Yes [1]."
        assert s["rationale"] == "Because [1]."

    def test_case_insensitive_and_multiline_bodies(self):
        text = (
            "short_answer: First line.\nStill the short answer.\n"
            "LIMITATIONS: none"
        )
        s = _parse_answer_sections(text)
        assert "Still the short answer." in s["short_answer"]
        assert s["limitations"] == "none"
        assert "recommended_action" not in s

    def test_freeform_text_yields_nothing(self):
        assert _parse_answer_sections("Just a plain paragraph answer.") == {}
