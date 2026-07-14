"""Deterministic unit tests for the chat endpoint helpers (no Ollama needed)."""



from api.prompts import _parse_answer_sections


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


class TestGermanAnswers:
    """The corpus and the question are German; the labels are English.

    The labels are structure, not content — the model writes prose under them in
    the language it was asked in. Nothing may key off the prose being English.
    """

    def test_german_prose_under_english_labels(self):
        text = (
            "SHORT_ANSWER: Bei MRSA-Kolonisation genügt Standardhygiene [1].\n"
            "RECOMMENDED_ACTION: Handschuhe und Schutzkittel bei Kontakt anlegen [1].\n"
            "RATIONALE: Übertragung erfolgt überwiegend über die Hände [1][2].\n"
            "LIMITATIONS: Die Quellen decken die Sanierung nicht ab."
        )
        sections = _parse_answer_sections(text)
        assert sections["short_answer"] == "Bei MRSA-Kolonisation genügt Standardhygiene [1]."
        assert sections["recommended_action"].startswith("Handschuhe")
        assert "Hände" in sections["rationale"]
        assert "Sanierung" in sections["limitations"]

    def test_umlauts_and_eszett_survive(self):
        sections = _parse_answer_sections("SHORT_ANSWER: Schutzmaßnahmen für Räume [1].")
        assert sections["short_answer"] == "Schutzmaßnahmen für Räume [1]."

    def test_citation_markers_are_found_in_german_text(self):
        # The grounding verdict counts [n] in the answer, whatever language it is.
        import re
        text = "SHORT_ANSWER: Einzelzimmer erforderlich [1][2]."
        assert {int(n) for n in re.findall(r"\[(\d+)\]", text)} == {1, 2}
