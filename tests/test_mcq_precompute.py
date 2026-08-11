"""
Phase 5 — multiple-choice pre-filter.

Every leaf is judged against every question ONCE, at index time, keyed by the
leaf's content hash. At query time the selected answers' leaf sets are unioned
(more selections = wider candidate set, by design) and the ancestor closure of
that leaf set becomes the tree the user's query runs over.
"""
import json

import pytest

import choices
import config as app_config
import pageindex

pytestmark = pytest.mark.unit


def _leaf(node_id, content):
    return pageindex.PageNode(node_id=node_id, title=node_id.title(), heading_level=3,
                              line_idx=0, summary="s", content=content)


QUESTIONS = choices.QuestionSet(version=1, questions=[
    choices.Question(id="q1", prompt="Setting?", answers=[
        choices.Answer("q1__icu", "ICU", "intensive care"),
        choices.Answer("q1__ward", "Ward", "general ward"),
    ]),
    choices.Question(id="q2", prompt="Population?", answers=[
        choices.Answer("q2__adult", "Adults", "adult patients"),
        choices.Answer("q2__child", "Children", "paediatric patients"),
    ]),
])


class Chat:
    """Scripted facet responses, counting calls."""

    def __init__(self, mapping=None):
        self.calls = []
        self.mapping = mapping or {}

    def __call__(self, prompt):
        self.calls.append(prompt)
        for needle, answers in self.mapping.items():
            if needle in prompt:
                return json.dumps({"answers": answers})
        return json.dumps({"answers": []})


def _precompute(tmp_path, leaves, chat, model="m1", questions=QUESTIONS):
    return choices.precompute_document(
        "doc", leaves, tmp_path, chat, json.loads, model, questions=questions)


class TestCost:
    def test_one_call_per_leaf_per_question_not_per_answer(self, tmp_path):
        leaves = [_leaf("a", "x"), _leaf("b", "y"), _leaf("c", "z")]
        chat = Chat()
        report = _precompute(tmp_path, leaves, chat)

        # 3 leaves x 2 questions = 6, NOT 3 x 4 answers = 12.
        assert len(chat.calls) == 6
        assert report["calls"] == 6

    def test_a_question_with_many_answers_still_costs_one_call(self, tmp_path):
        wide = choices.QuestionSet(version=1, questions=[
            choices.Question(id="q", prompt="?", answers=[
                choices.Answer(f"q__a{i}", f"A{i}", f"facet {i}") for i in range(8)])])
        chat = Chat()
        _precompute(tmp_path, [_leaf("a", "x")], chat, questions=wide)
        assert len(chat.calls) == 1

    def test_answers_are_offered_together_in_one_prompt(self, tmp_path):
        chat = Chat()
        _precompute(tmp_path, [_leaf("a", "x")], chat)
        assert "id=q1__icu" in chat.calls[0] and "id=q1__ward" in chat.calls[0]

    def test_prompt_uses_the_facet_not_the_label(self, tmp_path):
        """label is for the user, facet is what the model judges against."""
        chat = Chat()
        _precompute(tmp_path, [_leaf("a", "x")], chat)
        assert "intensive care" in chat.calls[0]


class TestPersistence:
    def test_rerun_on_unchanged_leaves_costs_nothing(self, tmp_path):
        leaves = [_leaf("a", "x"), _leaf("b", "y")]
        _precompute(tmp_path, leaves, Chat())

        chat = Chat()
        report = _precompute(tmp_path, leaves, chat)
        assert len(chat.calls) == 0
        assert report["reused"] == 4

    def test_only_the_edited_leaf_is_recomputed(self, tmp_path):
        leaves = [_leaf("a", "x"), _leaf("b", "y")]
        _precompute(tmp_path, leaves, Chat())

        chat = Chat()
        edited = [_leaf("a", "x"), _leaf("b", "COMPLETELY DIFFERENT")]
        report = _precompute(tmp_path, edited, chat)
        assert len(chat.calls) == 2, "one call per question, for the edited leaf only"
        assert report["reused"] == 2

    def test_judgements_survive_a_node_id_change(self, tmp_path):
        """Keyed by the passage itself (title + content), not by node_id — so
        a re-index that renumbers ids, e.g. after a heading is inserted above,
        reuses every judgement instead of paying for the document again."""
        def leaf_with_id(node_id):
            return pageindex.PageNode(node_id=node_id, title="Imaging", heading_level=3,
                                      line_idx=0, summary="s", content="stable content")

        _precompute(tmp_path, [leaf_with_id("imaging")], Chat())
        chat = Chat()
        report = _precompute(tmp_path, [leaf_with_id("imaging-2")], chat)
        assert len(chat.calls) == 0
        assert report["reused"] == 2

    def test_a_changed_title_does_invalidate(self, tmp_path):
        """The title is part of the passage the model judges, so it counts."""
        def leaf_titled(title):
            return pageindex.PageNode(node_id="x", title=title, heading_level=3,
                                      line_idx=0, summary="s", content="stable content")

        _precompute(tmp_path, [leaf_titled("Imaging")], Chat())
        chat = Chat()
        _precompute(tmp_path, [leaf_titled("Radiology")], chat)
        assert len(chat.calls) == 2

    def test_model_change_invalidates_everything(self, tmp_path):
        leaves = [_leaf("a", "x")]
        _precompute(tmp_path, leaves, Chat(), model="m1")
        chat = Chat()
        _precompute(tmp_path, leaves, chat, model="m2")
        assert len(chat.calls) == 2

    def test_prompt_version_bump_invalidates_everything(self, tmp_path, monkeypatch):
        leaves = [_leaf("a", "x")]
        _precompute(tmp_path, leaves, Chat())

        bumped = dict(app_config.PROMPT_VERSIONS)
        bumped["facet_leaf"] += 1
        monkeypatch.setattr(app_config, "PROMPT_VERSIONS", bumped)

        chat = Chat()
        _precompute(tmp_path, leaves, chat)
        assert len(chat.calls) == 2

    def test_question_set_version_bump_invalidates_everything(self, tmp_path):
        leaves = [_leaf("a", "x")]
        _precompute(tmp_path, leaves, Chat())
        v2 = choices.QuestionSet(version=2, questions=QUESTIONS.questions)
        chat = Chat()
        _precompute(tmp_path, leaves, chat, questions=v2)
        assert len(chat.calls) == 2

    def test_deleted_leaves_are_pruned_from_the_store(self, tmp_path):
        _precompute(tmp_path, [_leaf("a", "x"), _leaf("b", "y")], Chat())
        _precompute(tmp_path, [_leaf("a", "x")], Chat())
        store = choices.FacetStore(tmp_path, "doc")
        assert len(store.data) == 1


class TestRobustness:
    def test_hallucinated_answer_ids_are_dropped(self, tmp_path):
        chat = Chat({"PASSAGE": ["q1__icu", "not-a-real-answer"]})
        _precompute(tmp_path, [_leaf("a", "x")], chat)
        store = choices.FacetStore(tmp_path, "doc")
        stored = store.get(choices.leaf_hash("A", "x"), "q1")
        assert stored == ["q1__icu"]

    def test_a_failed_judgement_is_recall_biased(self, tmp_path):
        """Exclusion here is unrecoverable, so a failure must include the leaf
        everywhere rather than drop it from every answer."""
        def broken(prompt):
            raise ConnectionError("down")

        report = choices.precompute_document(
            "doc", [_leaf("a", "x")], tmp_path, broken, json.loads, "m1",
            questions=QUESTIONS)

        assert report["errors"]
        store = choices.FacetStore(tmp_path, "doc")
        stored = store.get(choices.leaf_hash("A", "x"), "q1")
        assert set(stored) == {"q1__icu", "q1__ward"}


class TestSelection:
    def _store(self, tmp_path):
        chat = Chat({
            "icu passage": ["q1__icu", "q2__adult"],
            "ward passage": ["q1__ward"],
            "child passage": ["q2__child"],
        })
        leaves = [_leaf("a", "icu passage"), _leaf("b", "ward passage"),
                  _leaf("c", "child passage"), _leaf("d", "unrelated")]
        _precompute(tmp_path, leaves, chat)
        return choices.FacetStore(tmp_path, "doc"), leaves

    def test_single_answer_selects_its_leaves(self, tmp_path):
        store, leaves = self._store(tmp_path)
        assert choices.selected_leaf_ids(store, leaves, ["q1__icu"]) == {"a"}

    def test_selections_union_rather_than_intersect(self, tmp_path):
        store, leaves = self._store(tmp_path)
        got = choices.selected_leaf_ids(store, leaves, ["q1__ward", "q2__child"])
        assert got == {"b", "c"}, "union — each answer contributes its own leaves"

    def test_more_selections_never_narrow(self, tmp_path):
        store, leaves = self._store(tmp_path)
        one = choices.selected_leaf_ids(store, leaves, ["q1__icu"])
        two = choices.selected_leaf_ids(store, leaves, ["q1__icu", "q1__ward"])
        assert one <= two

    def test_order_independent_and_idempotent(self, tmp_path):
        store, leaves = self._store(tmp_path)
        a = choices.selected_leaf_ids(store, leaves, ["q1__icu", "q2__child"])
        b = choices.selected_leaf_ids(store, leaves, ["q2__child", "q1__icu"])
        c = choices.selected_leaf_ids(store, leaves, ["q1__icu", "q1__icu", "q2__child"])
        assert a == b == c

    def test_no_selection_means_everything(self, tmp_path):
        store, leaves = self._store(tmp_path)
        assert choices.selected_leaf_ids(store, leaves, []) == {"a", "b", "c", "d"}

    def test_unmatched_selection_yields_nothing(self, tmp_path):
        store, leaves = self._store(tmp_path)
        assert choices.selected_leaf_ids(store, leaves, ["q2__adult"]) == {"a"}
        assert choices.selected_leaf_ids(store, leaves, ["q_nonexistent"]) == set()

    def test_tags_say_which_answers_put_a_leaf_in_play(self, tmp_path):
        store, leaves = self._store(tmp_path)
        icu_leaf = next(l for l in leaves if l.node_id == "a")
        tags = choices.answers_for_leaf(store, icu_leaf, ["q1__icu", "q2__child"])
        assert tags == ["q1__icu"], (
            "under union, membership differs per leaf — that is the signal")


class TestClosure:
    def _tree(self):
        return [pageindex.PageNode(
            node_id="root", title="Root", heading_level=1, line_idx=0, summary="",
            children=[
                pageindex.PageNode(node_id="s1", title="S1", heading_level=2,
                                   line_idx=1, summary="", children=[_leaf("a", "x")]),
                pageindex.PageNode(node_id="s2", title="S2", heading_level=2,
                                   line_idx=2, summary="", children=[_leaf("b", "y")]),
            ])]

    def test_closure_includes_every_ancestor(self):
        keep = choices.ancestor_closure(self._tree(), {"a"})
        assert keep == {"root", "s1", "a"}

    def test_closure_excludes_unrelated_branches(self):
        assert "s2" not in choices.ancestor_closure(self._tree(), {"a"})

    def test_pruned_tree_is_connected_and_complete(self):
        pruned = choices.prune_tree_to(self._tree(), choices.ancestor_closure(
            self._tree(), {"a"}))
        assert len(pruned) == 1
        assert [c.node_id for c in pruned[0].children] == ["s1"]
        assert [l.node_id for l in pageindex._collect_leaves(pruned)] == ["a"]

    def test_original_tree_is_not_mutated(self):
        tree = self._tree()
        choices.prune_tree_to(tree, {"root", "s1", "a"})
        assert len(tree[0].children) == 2, "pruning must copy, not mutate the index"

    def test_empty_selection_yields_empty_tree(self):
        assert choices.prune_tree_to(self._tree(), set()) == []


class TestQuestionConfig:
    def test_placeholders_load_when_no_file_exists(self, tmp_path):
        qs = choices.load_questions(tmp_path / "missing.json")
        assert qs.questions and all(q.answers for q in qs.questions)

    def test_roundtrip_through_disk(self, tmp_path):
        p = tmp_path / "q.json"
        choices.write_default_questions(p)
        loaded = choices.load_questions(p)
        assert loaded.to_dict() == choices.DEFAULT_QUESTIONS.to_dict()

    def test_malformed_file_falls_back_with_a_warning(self, tmp_path, capsys):
        p = tmp_path / "q.json"
        p.write_text("{ not json")
        qs = choices.load_questions(p)
        assert qs.questions
        assert "could not read" in capsys.readouterr().err

    def test_label_and_facet_are_separate(self):
        answer = choices.DEFAULT_QUESTIONS.questions[0].answers[0]
        assert answer.label and answer.facet and answer.label != answer.facet

    def test_answer_ids_are_unique(self):
        ids = [a.id for q in choices.DEFAULT_QUESTIONS.questions for a in q.answers]
        assert len(ids) == len(set(ids))


class TestIntegrationWithPruning:
    def test_selection_narrows_the_tree_before_any_llm_call(
            self, tmp_corpus, fake_ollama, monkeypatch):
        monkeypatch.setattr(choices, "QUESTIONS_PATH", tmp_corpus["root"] / "q.json")
        choices.reset_cache()

        pageindex.build_index(tmp_corpus["doc"], use_llm_summaries=False)
        nodes = pageindex.load_index_nodes(tmp_corpus["doc"])
        leaves = pageindex._collect_leaves(nodes)

        # Only the imaging leaf is relevant to q1__icu.
        chat = Chat({"CT is preferred": ["q1__icu"]})
        choices.precompute_document(tmp_corpus["doc"], leaves, tmp_corpus["index"],
                                    chat, json.loads, "m1", questions=QUESTIONS)

        fake_ollama.reset()
        fake_ollama.default_response = json.dumps({"keep": []})
        ctx = pageindex.new_run()
        doc = pageindex.prune_document(tmp_corpus["doc"], "query", ctx=ctx,
                                       selected_answers=["q1__icu"])

        kept = {l.node_id for l in doc.leaves}
        assert kept == {"imaging"}, f"expected only the imaging leaf, got {kept}"
        assert doc.choice_tags["imaging"] == ["q1__icu"]

    def test_empty_match_falls_back_to_the_whole_document(
            self, tmp_corpus, fake_ollama, monkeypatch, capsys):
        monkeypatch.setattr(choices, "QUESTIONS_PATH", tmp_corpus["root"] / "q.json")
        choices.reset_cache()
        pageindex.build_index(tmp_corpus["doc"], use_llm_summaries=False)

        fake_ollama.default_response = json.dumps({"keep": []})
        ctx = pageindex.new_run()
        doc = pageindex.prune_document(tmp_corpus["doc"], "query", ctx=ctx,
                                       selected_answers=["q1__icu"])

        assert len(doc.leaves) > 0, "must not silently return an empty corpus"
        assert "no leaves matched" in capsys.readouterr().err
