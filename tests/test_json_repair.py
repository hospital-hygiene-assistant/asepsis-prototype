"""The model's JSON is repaired rather than discarded.

The leaf evaluator asks for a VERBATIM quote, so a passage containing a
quotation mark makes the model emit unescaped `"` inside a string value. That
cost two evaluation attempts per query touching such a passage before
_repair_json existed.
"""
import pytest

from pageindex import _parse_json_response, _repair_json


class TestUnescapedQuotes:
    def test_stray_quotes_inside_a_value_are_escaped(self):
        raw = ('{"relevant": true, "quote": "referred to as the "global survey" '
               'among experts."}')
        assert _parse_json_response(raw)["quote"].endswith('among experts.')
        assert '"global survey"' in _parse_json_response(raw)["quote"]

    def test_a_closing_quote_is_still_a_closing_quote(self):
        raw = '{"relevant": true, "reason": "fine", "quote": "clean"}'
        assert _parse_json_response(raw) == {
            "relevant": True, "reason": "fine", "quote": "clean"}


class TestTruncation:
    def test_cut_off_mid_string_keeps_what_arrived(self):
        raw = '{"relevant": true, "quote": "An estimated 4.95 million deaths'
        assert "4.95 million" in _parse_json_response(raw)["quote"]

    def test_cut_off_before_the_closing_brace(self):
        assert _parse_json_response('{"relevant": false') == {"relevant": False}

    def test_open_containers_are_balanced_in_order(self):
        assert _repair_json('{"a": [1, 2').endswith("]}")


class TestRegressions:
    @pytest.mark.parametrize("raw,expected", [
        ('```json\n{"relevant": false}\n```', {"relevant": False}),
        ('“relevant”: true' .join(("{", "}")), {"relevant": True}),
        ('{"relevant": true, "reason": "one\ntwo"}',
         {"relevant": True, "reason": "one\ntwo"}),
    ])
    def test_existing_shapes_still_parse(self, raw, expected):
        assert _parse_json_response(raw) == expected
