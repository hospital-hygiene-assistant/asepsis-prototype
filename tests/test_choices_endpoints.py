"""The server side of the multiple-choice pre-filter.

These are the seams where the filter's effect either reaches the user or
disappears silently — the cache key, the corpus-level report, and the cached
question set. Every one of them was previously a place where a narrowed search
looked identical to an unnarrowed one.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tauri-app"))

import choices as app_choices  # noqa: E402
from server import _cache_key_for, _choice_payload  # noqa: E402

pytestmark = pytest.mark.unit


class TestCacheKey:
    """The mode is part of the retrieval, so it must be part of the key.

    Without it, asking the same question in Fast mode replays the wider run's
    answer — or worse, a Fast-mode answer is served to someone who did not ask
    for one.
    """

    def test_the_two_modes_do_not_share_a_key(self):
        answers = ["q_setting__icu", "q_stage__treat"]
        assert (_cache_key_for("q", None, answers, app_choices.ANY)
                != _cache_key_for("q", None, answers, app_choices.ALL))

    def test_the_same_mode_is_stable(self):
        answers = ["q_setting__icu"]
        assert (_cache_key_for("q", None, answers, app_choices.ALL)
                == _cache_key_for("q", None, answers, app_choices.ALL))

    def test_no_answers_means_the_mode_is_irrelevant(self):
        """With nothing selected the filter does nothing, so the two modes ARE
        the same retrieval and should share the cached run."""
        assert (_cache_key_for("q", None, [], app_choices.ANY)
                == _cache_key_for("q", None, [], app_choices.ALL))

    def test_different_answers_still_differ(self):
        assert (_cache_key_for("q", None, ["q_setting__icu"])
                != _cache_key_for("q", None, ["q_setting__ward"]))


class TestChoicePayload:
    """The corpus-level account shown above an answer."""

    def _results(self, *filters):
        return {f"doc{i}": {"choice_filter": f} for i, f in enumerate(filters)}

    def test_absent_when_nothing_was_selected(self):
        assert _choice_payload(self._results({}, {}))["applied"] is False

    def test_counts_are_summed_across_documents(self):
        payload = _choice_payload(self._results(
            {"applied": True, "doc": "a", "state": "filtered", "kept": 3, "total": 10},
            {"applied": True, "doc": "b", "state": "filtered", "kept": 2, "total": 8},
        ), ["q_setting__icu"])
        assert (payload["kept"], payload["total"]) == (5, 18)

    def test_an_unfiltered_document_is_named_not_just_counted(self):
        """A document searched whole for want of judgements is a coverage hole.
        Naming it is the difference between a fixable gap and a silent one."""
        payload = _choice_payload(self._results(
            {"applied": True, "doc": "new_pdf", "state": "unfiltered",
             "kept": 9, "total": 9},
            {"applied": True, "doc": "old", "state": "filtered", "kept": 2, "total": 8},
        ), ["q_setting__icu"])
        assert payload["docs_unfiltered"] == ["new_pdf"]

    def test_excluded_documents_are_counted(self):
        payload = _choice_payload(self._results(
            {"applied": True, "doc": "a", "state": "excluded", "kept": 0, "total": 7},
            {"applied": True, "doc": "b", "state": "filtered", "kept": 2, "total": 8},
        ), ["q_setting__icu"])
        assert payload["docs_excluded"] == 1

    def test_the_mode_is_carried_through(self):
        payload = _choice_payload(self._results(
            {"applied": True, "doc": "a", "state": "filtered", "kept": 1, "total": 4}),
            ["q_setting__icu"], app_choices.ALL)
        assert payload["mode"] == app_choices.ALL


class TestShippedQuestionSet:
    def test_the_config_file_is_what_loads(self):
        """Not the in-code fallback — a typo in the JSON would otherwise be
        invisible, since the loader falls back silently by design."""
        app_choices.reset_cache()
        qs = app_choices.load_questions()
        assert qs.to_dict() != app_choices.DEFAULT_QUESTIONS.to_dict()
        assert app_choices.QUESTIONS_PATH.exists()

    def test_every_answer_resolves_to_its_question(self):
        qs = app_choices.load_questions()
        for question in qs.questions:
            for answer in question.answers:
                assert qs.question_of(answer.id) is question

    def test_answers_group_by_question(self):
        qs = app_choices.load_questions()
        picked = [qs.questions[0].answers[0].id, qs.questions[1].answers[0].id]
        assert len(app_choices.group_by_question(picked, qs)) == 2


class TestIndexModuleContract:
    """The server selects its retrieval path with hasattr.

    `pageindex_custom` re-exports an explicit list of names from pageindex, and
    `prune_document`/`evaluate_ranked` were not on it. The server silently took
    the single-document fallback instead — which never passes `selected_answers`
    — so the pre-filter filtered NOTHING in the running app, no matter how
    correct the code behind it was. hasattr checks fail silently by design, so
    the contract has to be asserted somewhere.
    """

    def _module(self):
        from modules.registry import defaults, load
        return load("index", defaults()["index"])

    @pytest.mark.parametrize("name", ["prune_document", "evaluate_ranked"])
    def test_the_two_phase_api_is_exported(self, name):
        assert hasattr(self._module(), name), (
            f"the default index module must export {name}, or the server "
            f"silently drops to the path that ignores the pre-filter")

    def test_prune_document_accepts_the_pre_filter_arguments(self):
        import inspect
        params = inspect.signature(self._module().prune_document).parameters
        assert "selected_answers" in params
        assert "selection_mode" in params
