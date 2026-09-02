"""Deterministic unit tests for the chat endpoint helpers (no Ollama needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tauri-app"))

from server import (  # noqa: E402
    _parse_answer_sections,
    _followup_context,
    _followup_frame,
    _with_context_source,
    _cited_numbers,
    USER_CONTEXT_DOC,
)
import server  # noqa: E402


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


def _run(passages="[1] doc > section\nSome evidence.", answer="An answer [1].",
         query="What now?", sources=None):
    return {"query": query, "passages": passages, "answer": answer,
            "sources": [{"n": 1}] if sources is None else sources,
            "thread": []}


def _turns(n, size=40):
    """n alternating turns, oldest first, each `size` characters."""
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"t{i} " + "x" * size} for i in range(n)]


class TestFollowupFrame:
    def test_frame_carries_passages_answer_and_question(self):
        frame = _followup_frame(_run())
        assert "Some evidence." in frame
        assert "An answer [1]." in frame
        assert "What now?" in frame

    def test_missing_answer_is_stated_not_blank(self):
        frame = _followup_frame({"query": "q", "passages": "p", "answer": ""})
        assert "not available" in frame


class TestFollowupContext:
    def test_short_thread_is_kept_whole(self):
        ctx = _followup_context(_run(), _turns(4))
        assert ctx["dropped_turns"] == 0
        assert ctx["turns"] == 4
        assert len(ctx["_kept"]) == 4

    def test_segments_sum_to_used(self):
        ctx = _followup_context(_run(), _turns(3))
        seg = ctx["segments"]
        total = sum(seg[k] for k in
                    ("instructions", "passages", "answer", "conversation"))
        assert total == ctx["used"]

    def test_overflow_drops_oldest_and_reports_it(self):
        # A frame that alone nearly fills the window leaves room for a turn or
        # two at most, so the rest must be dropped rather than silently sent.
        big = _run(passages="p " * 200_000)
        ctx = _followup_context(big, _turns(6))
        assert ctx["dropped_turns"] > 0
        assert ctx["turns"] + ctx["dropped_turns"] == 6

    def test_survivors_are_the_most_recent_turns(self):
        big = _run(passages="p " * 120_000)
        thread = _turns(8, size=4000)
        ctx = _followup_context(big, thread)
        assert ctx["dropped_turns"] > 0
        # Whatever survived is a SUFFIX of the thread, in original order.
        assert ctx["_kept"] == thread[len(thread) - len(ctx["_kept"]):]

    def test_passages_are_never_dropped_to_make_room(self):
        # Citations only mean anything while every passage is present, so the
        # frame is pinned even when the thread cannot fit at all.
        big = _run(passages="p " * 400_000)
        ctx = _followup_context(big, _turns(4))
        assert ctx["segments"]["passages"] > 0
        assert "p p" in ctx["_frame"]
        assert ctx["dropped_turns"] == 4

    def test_reply_reserve_is_held_back_from_the_budget(self):
        ctx = _followup_context(_run(), [])
        assert ctx["budget"] == ctx["max"] - ctx["reserve"]
        assert ctx["reserve"] > 0


class TestEmptyRetrievalFollowup:
    """A search that found nothing is still a run worth talking about — but it
    must be talked about under different instructions, or the model answers
    the clinical question from its own training."""

    def test_empty_run_gets_the_no_evidence_frame(self):
        frame = _followup_frame(_run(passages="", answer="Nothing matched.",
                                     sources=[]))
        assert "NO passages" in frame
        assert "must not answer" in frame
        assert "own knowledge" in frame

    def test_empty_run_still_yields_a_usable_context(self):
        ctx = _followup_context(_run(passages="", answer="Nothing matched.",
                                     sources=[]), _turns(2))
        assert ctx["empty_retrieval"] is True
        assert ctx["used"] > 0
        assert ctx["dropped_turns"] == 0

    def test_populated_run_is_not_flagged_empty(self):
        assert _followup_context(_run(), [])["empty_retrieval"] is False


class TestContextAsASource:
    """Context the clinician types is cited by the rewritten answer, so it has
    to be a numbered source like any other — an answer leaning on something
    with no card behind it cannot be checked."""

    def _two_source_run(self):
        return {"sources": [{"n": 1, "id": "s1", "doc": "a", "title": "A"},
                            {"n": 2, "id": "s2", "doc": "b", "title": "B"}],
                "passages": "[1] a\nfirst\n\n[2] b\nsecond"}

    def test_context_is_appended_as_the_next_number(self):
        sources, block, n = _with_context_source(self._two_source_run(), "pt is 82")
        assert n == 3
        assert sources[-1]["n"] == 3
        assert sources[-1]["user_context"] is True
        assert "[3] Context supplied by the clinician" in block
        assert "pt is 82" in block

    def test_document_passages_keep_their_numbers(self):
        sources, block, _ = _with_context_source(self._two_source_run(), "x")
        assert [s["n"] for s in sources[:2]] == [1, 2]
        assert block.startswith("[1] a")
        assert "[2] b" in block

    def test_rewriting_twice_replaces_rather_than_stacks(self):
        run = self._two_source_run()
        sources, block, n = _with_context_source(run, "first context")
        again, block2, n2 = _with_context_source(
            {"sources": sources, "passages": block}, "second context")
        assert n2 == n == 3
        assert len([s for s in again if s.get("doc") == USER_CONTEXT_DOC]) == 1
        assert "second context" in block2
        assert "first context" not in block2

    def test_works_when_nothing_was_retrieved(self):
        sources, block, n = _with_context_source(
            {"sources": [], "passages": ""}, "pt is 82")
        assert n == 1
        assert len(sources) == 1
        assert block.startswith("[1] Context supplied by the clinician")


class TestRunAddressing:
    """An answer must stay reachable after later questions have been asked.
    With a single slot, rewriting answer 1 silently used answer 3's passages."""

    def setup_method(self):
        server._runs.clear()
        server._active_run_id = ""
        server._run_seq = 0

    def _seed(self, n):
        ids = []
        for i in range(n):
            rid = server._new_run_id()
            server._runs[rid] = {"query": f"q{i}", "sources": [{"n": 1}],
                                 "passages": f"p{i}", "answer": f"a{i}",
                                 "thread": []}
            server._active_run_id = rid
            ids.append(rid)
        return ids

    def test_ids_are_distinct(self):
        assert len(set(self._seed(3))) == 3

    def test_bare_lookup_returns_the_active_run(self):
        ids = self._seed(3)
        rid, run = server._get_run(None)
        assert rid == ids[-1]
        assert run["query"] == "q2"

    def test_an_earlier_run_is_still_addressable(self):
        ids = self._seed(3)
        rid, run = server._get_run(ids[0])
        assert rid == ids[0]
        assert run["passages"] == "p0"

    def test_unknown_id_is_not_silently_the_active_one(self):
        self._seed(2)
        assert server._get_run("r99") == ("", {})

    def test_no_runs_at_all_is_the_only_empty_case(self):
        assert server._get_run(None) == ("", {})

    def test_activate_moves_the_default(self):
        ids = self._seed(3)
        server.chat_activate(server.ActivateRequest(run_id=ids[0]))
        rid, _ = server._get_run(None)
        assert rid == ids[0]

    def test_threads_do_not_leak_between_runs(self):
        ids = self._seed(2)
        server._runs[ids[0]]["thread"] = [{"role": "user", "content": "only mine"}]
        assert server._runs[ids[1]]["thread"] == []


class TestRewritePrompt:
    """A rewrite has to BE a rewrite: the model needs the answer it is revising
    and explicit permission to change its verdict, or a good answer comes back
    verbatim with the clinician's context tacked on."""

    def _prompt(self, previous="SHORT_ANSWER: Give drug A."):
        return server.CHAT_REWRITE_PROMPT.format(
            previous=previous, sources="[1] doc\ntext\n\n[2] context",
            query="Which drug?", ctx_n=2)

    def test_the_previous_answer_is_in_the_prompt(self):
        assert "SHORT_ANSWER: Give drug A." in self._prompt()

    def test_changing_the_verdict_is_explicitly_allowed(self):
        p = self._prompt()
        assert "CHANGE THEM" in p
        assert "Do not restate the previous answer" in p

    def test_the_context_still_cannot_invent_evidence(self):
        assert "CANNOT do is create clinical evidence" in self._prompt()

    def test_what_changed_is_requested_and_parseable(self):
        assert "WHAT_CHANGED:" in self._prompt()
        parsed = _parse_answer_sections(
            "SHORT_ANSWER: Give drug B.\nWHAT_CHANGED: Switched from A to B [2].")
        assert parsed["what_changed"] == "Switched from A to B [2]."


class TestCitedNumbers:
    """Grouped citations come back in two shapes, and counting only one of
    them marks a well-cited answer as ungrounded."""

    def test_separate_brackets(self):
        assert _cited_numbers("a [1] b [2][3]") == [1, 2, 3]

    def test_comma_grouped(self):
        assert _cited_numbers("a [2, 3] b") == [2, 3]

    def test_mixed_and_spaced(self):
        assert sorted(set(_cited_numbers("[1] and [2,3] and [4, 5][6]"))) == [1, 2, 3, 4, 5, 6]

    def test_no_citations(self):
        assert _cited_numbers("plain prose") == []
        assert _cited_numbers("") == []

    def test_non_numeric_brackets_are_ignored(self):
        assert _cited_numbers("[see note] and [1]") == [1]
