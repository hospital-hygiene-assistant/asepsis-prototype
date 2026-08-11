"""
Phase 0 — error handling.

The old behaviour was asymmetric and lossy: a failed section check kept the
section (fine), but a failed *leaf* evaluation was silently recorded as
"rejected" — a transport hiccup deleted evidence and looked exactly like the
model judging the passage irrelevant. `error` is now its own status.
"""
import pytest

import pageindex

pytestmark = pytest.mark.unit


def _leaf(node_id="l1", content="Some clinical content."):
    return pageindex.PageNode(node_id=node_id, title="Leaf", heading_level=2,
                              line_idx=0, summary="", content=content)


def _section(children=None):
    return pageindex.PageNode(node_id="s1", title="Section", heading_level=1,
                              line_idx=0, summary="", children=children or [_leaf()])


class TestLeafEvalErrors:
    def test_transport_failure_is_error_not_rejected(self, fake_ollama, monkeypatch):
        def boom(*a, **kw):
            raise ConnectionError("ollama is down")
        monkeypatch.setattr(pageindex, "_chat", boom)

        ctx = pageindex.new_run()
        _, meta = pageindex._evaluate_leaf(_leaf(), "q", "doc", "crumb", "", ctx=ctx)

        assert meta["status"] == "error"
        assert meta["relevant"] is False
        assert "ollama is down" in meta["reason"]
        assert ctx.events_snapshot()["error"] == ["l1"]
        assert ctx.events_snapshot()["rejected"] == [], (
            "a failed call must never be recorded as a relevance judgement")

    def test_unparseable_output_retries_then_errors(self, fake_ollama):
        fake_ollama.default_response = "I'm afraid I can't do that, Dave."
        ctx = pageindex.new_run()
        _, meta = pageindex._evaluate_leaf(_leaf(), "q", "doc", "crumb", "", ctx=ctx)

        assert fake_ollama.count == 2, "should retry once before giving up"
        assert "IMPORTANT" in fake_ollama.prompts[1], "retry must add the stricter instruction"
        assert meta["status"] == "error"

    def test_retry_succeeds_after_a_bad_first_response(self, fake_ollama):
        fake_ollama.scripted = [
            "here you go:",
            '{"relevant": true, "reason": "it answers", "quote": "verbatim"}',
        ]
        ctx = pageindex.new_run()
        _, meta = pageindex._evaluate_leaf(_leaf(), "q", "doc", "crumb", "", ctx=ctx)

        assert fake_ollama.count == 2
        assert meta["status"] == "retrieved"
        assert meta["quote"] == "verbatim"

    def test_clean_rejection_is_still_rejected(self, fake_ollama):
        fake_ollama.default_response = '{"relevant": false, "reason": "off topic"}'
        ctx = pageindex.new_run()
        _, meta = pageindex._evaluate_leaf(_leaf(), "q", "doc", "crumb", "", ctx=ctx)

        assert fake_ollama.count == 1, "a clean answer must not trigger a retry"
        assert meta["status"] == "rejected"
        assert meta["reason"] == "off topic"

    def test_quote_is_dropped_on_rejection(self, fake_ollama):
        fake_ollama.default_response = (
            '{"relevant": false, "reason": "no", "quote": "should not survive"}')
        _, meta = pageindex._evaluate_leaf(_leaf(), "q", "doc", "crumb", "")
        assert meta["quote"] == ""


class TestSectionCheckErrors:
    def test_failure_keeps_the_section_but_records_error(self, fake_ollama, monkeypatch):
        def boom(*a, **kw):
            raise ConnectionError("nope")
        monkeypatch.setattr(pageindex, "_chat", boom)

        ctx = pageindex.new_run()
        verdict, reason = pageindex._check_section_relevant(_section(), "q", "crumb", ctx=ctx)

        assert verdict is True, "never prune on error — that silently drops content"
        assert ctx.events_snapshot()["error"] == ["s1"]
        assert "nope" in reason

    def test_clean_keep_is_recorded_as_kept(self, fake_ollama):
        fake_ollama.default_response = '{"relevant": true, "reason": "plausible"}'
        ctx = pageindex.new_run()
        verdict, _ = pageindex._check_section_relevant(_section(), "q", "crumb", ctx=ctx)
        assert verdict is True
        assert ctx.events_snapshot()["kept"] == ["s1"]
        assert ctx.events_snapshot()["error"] == []


class TestDeadCodeRemoved:
    def test_flatten_toc_is_gone(self):
        assert not hasattr(pageindex, "_flatten_toc")

    def test_find_nodes_by_ids_is_gone(self):
        assert not hasattr(pageindex, "_find_nodes_by_ids")
