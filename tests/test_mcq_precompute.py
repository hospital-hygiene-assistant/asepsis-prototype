"""
Phase 5 — multiple-choice pre-filter.

Every leaf is judged against every question ONCE, at index time, keyed by the
leaf's content hash. At query time the selected answers pick a leaf set and the
ancestor closure of it becomes the tree the user's query runs over.

Two selection modes, and which one is the DEFAULT is a clinical decision, not
a technical one:

  ANY (default)  a leaf matching any selected answer is a candidate. More
                 answers = wider search. Recall-first, because a passage
                 dropped here can never reach the answer.
  ALL (Fast)     a leaf must satisfy every question answered. Much narrower
                 and much faster, and it will drop passages a clinician might
                 have wanted — so it is opt-in and never inferred.
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


def _precompute(tmp_path, leaves, chat, model="m1", questions=QUESTIONS, **kw):
    return choices.precompute_document(
        "doc", leaves, tmp_path, chat, json.loads, model, questions=questions, **kw)


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


def _wire_questions(tmp_corpus, monkeypatch):
    """Point the loader at a real file holding the fixture questions.

    Retrieval resolves the question set from disk and keys stored judgements
    on its content, so a test that precomputes with one set and queries with
    another is testing the stale-store guard, not the pre-filter.
    """
    path = tmp_corpus["root"] / "q.json"
    path.write_text(json.dumps(QUESTIONS.to_dict()), encoding="utf-8")
    monkeypatch.setattr(choices, "QUESTIONS_PATH", path)
    choices.reset_cache()


class TestIntegrationWithPruning:
    def test_selection_narrows_the_tree_before_any_llm_call(
            self, tmp_corpus, fake_ollama, monkeypatch):
        _wire_questions(tmp_corpus, monkeypatch)

        pageindex.build_index(tmp_corpus["doc"], use_llm_summaries=False)
        nodes = pageindex.load_index_nodes(tmp_corpus["doc"])
        leaves = pageindex._collect_leaves(nodes)

        # Only the imaging leaf is relevant to q1__icu.
        chat = Chat({"CT is preferred": ["q1__icu"]})
        choices.precompute_document(tmp_corpus["doc"], leaves, tmp_corpus["index"],
                                    chat, json.loads, pageindex.MODEL,
                                    questions=QUESTIONS)

        fake_ollama.reset()
        fake_ollama.default_response = json.dumps({"keep": []})
        ctx = pageindex.new_run()
        doc = pageindex.prune_document(tmp_corpus["doc"], "query", ctx=ctx,
                                       selected_answers=["q1__icu"])

        kept = {l.node_id for l in doc.leaves}
        assert kept == {"imaging"}, f"expected only the imaging leaf, got {kept}"
        assert doc.choice_tags["imaging"] == ["q1__icu"]

    def test_a_document_with_no_judgements_is_searched_whole_and_says_so(
            self, tmp_corpus, fake_ollama, monkeypatch, capsys):
        """The coverage hole that used to hide: a document ingested after the
        last precompute has no judgements, so it matches nothing, so it was
        searched whole — indistinguishable from having been filtered."""
        _wire_questions(tmp_corpus, monkeypatch)
        pageindex.build_index(tmp_corpus["doc"], use_llm_summaries=False)

        fake_ollama.default_response = json.dumps({"keep": []})
        ctx = pageindex.new_run()
        doc = pageindex.prune_document(tmp_corpus["doc"], "query", ctx=ctx,
                                       selected_answers=["q1__icu"])

        assert len(doc.leaves) > 0, "must not silently return an empty corpus"
        assert doc.choice_filter["state"] == "unfiltered"
        assert doc.choice_filter["unjudged"] == doc.choice_filter["total"]
        assert "no precomputed judgements" in capsys.readouterr().err

    def test_a_judged_document_that_matches_nothing_is_excluded(
            self, tmp_corpus, fake_ollama, monkeypatch, capsys):
        """Judged and irrelevant is a VERDICT, not a failure. Falling back to
        the whole document here defeated the entire point of asking."""
        _wire_questions(tmp_corpus, monkeypatch)
        pageindex.build_index(tmp_corpus["doc"], use_llm_summaries=False)
        leaves = pageindex._collect_leaves(
            pageindex.load_index_nodes(tmp_corpus["doc"]))
        # Judged against everything, relevant to nothing selected.
        choices.precompute_document(tmp_corpus["doc"], leaves, tmp_corpus["index"],
                                    Chat(), json.loads, pageindex.MODEL,
                                    questions=QUESTIONS)

        fake_ollama.reset()
        ctx = pageindex.new_run()
        doc = pageindex.prune_document(tmp_corpus["doc"], "query", ctx=ctx,
                                       selected_answers=["q1__icu"])

        assert doc.leaves == []
        assert doc.choice_filter["state"] == "excluded"
        assert "document excluded" in capsys.readouterr().err


class TestAnswerIdEcho:
    """The prompt lists answers as "- id=<id> | <facet>" and the model replies
    with the decoration attached. Matching the raw string against known ids
    dropped every one, so leaves stored as matching NOTHING and the pre-filter
    judged the whole library irrelevant to every answer — silently, because an
    empty result is indistinguishable from a confident negative.

    This is the same defect class as the pruning id echo. Fixed there first,
    and not carried across until it had shipped here too.
    """

    @pytest.mark.parametrize("echoed", [
        "id=q1__icu",
        "q1__icu",
        "id=q1__icu | intensive care",
        "- id=q1__icu",
        '"q1__icu"',
        "q1__icu | LEAF | intensive care",
    ])
    def test_every_echo_shape_resolves(self, echoed):
        valid = QUESTIONS.answer_ids()
        assert choices.normalise_answer_id(echoed, valid) == "q1__icu"

    def test_unrecognised_answers_are_dropped(self):
        assert choices.normalise_answer_id("totally-made-up", QUESTIONS.answer_ids()) is None

    def test_prefixed_ids_are_stored_not_discarded(self, tmp_path):
        chat = Chat({"PASSAGE": ["id=q1__icu", "id=q2__adult"]})
        _precompute(tmp_path, [_leaf("a", "icu passage")], chat)
        store = choices.FacetStore(tmp_path, "doc")
        assert store.get(choices.leaf_hash("A", "icu passage"), "q1") == ["q1__icu"]
        assert store.get(choices.leaf_hash("A", "icu passage"), "q2") == ["q2__adult"]

    def test_duplicates_collapse(self, tmp_path):
        chat = Chat({"PASSAGE": ["q1__icu", "id=q1__icu"]})
        _precompute(tmp_path, [_leaf("a", "x")], chat)
        store = choices.FacetStore(tmp_path, "doc")
        assert store.get(choices.leaf_hash("A", "x"), "q1") == ["q1__icu"]


class TestPrecomputeSanity:
    """A corpus-level smoke check on the OUTCOME, not the parsing.

    The parse bug was invisible per-leaf: every judgement looked like a valid
    confident negative. It was only obvious in aggregate — 65 of 67 leaves
    matching nothing at all. That shape is what to assert on."""

    def test_reports_when_almost_nothing_matches(self, tmp_path):
        chat = Chat()          # default: every answer list empty
        leaves = [_leaf(f"n{i}", f"passage {i}") for i in range(20)]
        report = _precompute(tmp_path, leaves, chat)

        store = choices.FacetStore(tmp_path, "doc")
        matched = sum(1 for judgements in store.data.values()
                      if any(judgements.values()))
        assert report["calls"] == 40
        assert matched == 0
        # The store must at least make this visible to a caller that looks.
        assert len(store.data) == 20, (
            "every leaf is recorded, so a caller can compute the match rate "
            "and notice a corpus-wide zero")


class TestFastMode:
    """ALL: OR within a question, AND across questions.

    The default stays ANY on purpose — on the real corpus a single answer like
    "adults" covers 54 of 67 leaves, so ANY with two answers searches nearly
    everything, while ALL cuts hard enough to drop passages a clinician might
    have wanted. That trade is the user's to make, so the only thing these
    tests really guard is that it is never made FOR them.
    """

    def _store(self, tmp_path):
        chat = Chat({
            "icu adult": ["q1__icu", "q2__adult"],
            "icu child": ["q1__icu", "q2__child"],
            "ward adult": ["q1__ward", "q2__adult"],
        })
        leaves = [_leaf("a", "icu adult"), _leaf("b", "icu child"),
                  _leaf("c", "ward adult"), _leaf("d", "unrelated")]
        _precompute(tmp_path, leaves, chat)
        return choices.FacetStore(tmp_path, "doc"), leaves

    def _ids(self, tmp_path, answers, mode):
        store, leaves = self._store(tmp_path)
        return choices.selected_leaf_ids(store, leaves, answers, QUESTIONS, mode)

    def test_across_questions_is_an_intersection(self, tmp_path):
        got = self._ids(tmp_path, ["q1__icu", "q2__adult"], choices.ALL)
        assert got == {"a"}, "icu AND adult — not icu OR adult"

    def test_within_a_question_is_still_a_union(self, tmp_path):
        got = self._ids(tmp_path, ["q2__adult", "q2__child"], choices.ALL)
        assert got == {"a", "b", "c"}, "answers to one question are alternatives"

    def test_all_is_never_wider_than_any(self, tmp_path):
        answers = ["q1__icu", "q2__adult"]
        assert (self._ids(tmp_path, answers, choices.ALL)
                <= self._ids(tmp_path, answers, choices.ANY))

    def test_the_default_is_the_recall_first_mode(self, tmp_path):
        store, leaves = self._store(tmp_path)
        answers = ["q1__icu", "q2__adult"]
        assert (choices.selected_leaf_ids(store, leaves, answers, QUESTIONS)
                == choices.selected_leaf_ids(store, leaves, answers, QUESTIONS,
                                             choices.ANY))

    @pytest.mark.parametrize("mode", ["", None, "ALL", "intersect", "fast"])
    def test_an_unrecognised_mode_falls_back_to_recall(self, mode):
        assert choices.normalise_mode(mode) == choices.ANY, (
            "a typo or a stale frontend must never silently narrow a clinical "
            "search")

    def test_one_answer_behaves_identically_in_both_modes(self, tmp_path):
        assert (self._ids(tmp_path, ["q1__icu"], choices.ALL)
                == self._ids(tmp_path, ["q1__icu"], choices.ANY) == {"a", "b"})

    def test_an_unknown_answer_id_satisfies_nothing(self, tmp_path):
        assert self._ids(tmp_path, ["q1__icu", "made__up"], choices.ALL) == set()

    def test_an_unjudged_leaf_survives_both_modes(self, tmp_path):
        """Recall bias holds even in Fast mode: never judged is not the same
        as judged irrelevant, and only one of those is evidence."""
        store, leaves = self._store(tmp_path)
        leaves.append(_leaf("new", "ingested after the precompute"))
        for mode in (choices.ANY, choices.ALL):
            selection = choices.select_leaves(store, leaves,
                                              ["q1__icu", "q2__adult"],
                                              QUESTIONS, mode)
            assert "new" in selection.kept
            assert selection.unjudged == {"new"}


class TestQuestionEdits:
    """The cache key hashes the question CONTENT.

    `version` is a hand-maintained integer, and hand-maintained integers get
    forgotten: editing a facet's wording used to leave every stored judgement
    in place, answering a question that no longer existed.
    """

    def _edited(self, **kw):
        answers = [choices.Answer("q1__icu", kw.get("label", "ICU"),
                                  kw.get("facet", "intensive care")),
                   choices.Answer("q1__ward", "Ward", "general ward")]
        return choices.QuestionSet(version=1, questions=[
            choices.Question(id="q1", prompt=kw.get("prompt", "Setting?"),
                             answers=answers)])

    def test_editing_a_facet_invalidates(self, tmp_path):
        _precompute(tmp_path, [_leaf("a", "x")], Chat(), questions=self._edited())
        chat = Chat()
        _precompute(tmp_path, [_leaf("a", "x")], chat,
                    questions=self._edited(facet="critically ill patients"))
        assert len(chat.calls) == 1, "a reworded facet is a different question"

    def test_editing_a_prompt_invalidates(self, tmp_path):
        _precompute(tmp_path, [_leaf("a", "x")], Chat(), questions=self._edited())
        chat = Chat()
        _precompute(tmp_path, [_leaf("a", "x")], chat,
                    questions=self._edited(prompt="Where is the patient?"))
        assert len(chat.calls) == 1

    def test_an_unchanged_set_still_reuses(self, tmp_path):
        _precompute(tmp_path, [_leaf("a", "x")], Chat(), questions=self._edited())
        chat = Chat()
        _precompute(tmp_path, [_leaf("a", "x")], chat, questions=self._edited())
        assert len(chat.calls) == 0

    def test_the_shipped_questions_are_well_formed(self):
        """The real config, not the fixtures: ids unique, facets distinct from
        labels, nothing empty."""
        qs = choices.load_questions()
        ids = [a.id for q in qs.questions for a in q.answers]
        assert len(ids) == len(set(ids))
        for q in qs.questions:
            assert q.prompt and q.answers
            for a in q.answers:
                assert a.label and a.facet and a.label != a.facet


class TestCoverage:
    """Precompute status is read from DISK.

    It used to live in a process variable, so a restart reported "not computed
    yet" over a fully populated store, and a document ingested since the last
    pass was never noticed at all.
    """

    def _coverage(self, tmp_path, leaves, model="m1"):
        return choices.document_coverage(tmp_path, "doc", leaves, model, QUESTIONS)

    def test_missing_store_reports_missing(self, tmp_path):
        assert self._coverage(tmp_path, [_leaf("a", "x")])["state"] == "missing"

    def test_a_completed_pass_reports_ready(self, tmp_path):
        leaves = [_leaf("a", "x"), _leaf("b", "y")]
        _precompute(tmp_path, leaves, Chat())
        cov = self._coverage(tmp_path, leaves)
        assert cov["state"] == "ready" and cov["judged"] == 2

    def test_a_document_grown_since_the_pass_reports_partial(self, tmp_path):
        leaves = [_leaf("a", "x")]
        _precompute(tmp_path, leaves, Chat())
        cov = self._coverage(tmp_path, leaves + [_leaf("b", "new section")])
        assert cov["state"] == "partial"
        assert (cov["judged"], cov["leaves"]) == (1, 2)

    def test_a_different_model_reports_stale(self, tmp_path):
        leaves = [_leaf("a", "x")]
        _precompute(tmp_path, leaves, Chat(), model="m1")
        assert self._coverage(tmp_path, leaves, model="m2")["state"] == "stale"

    def test_edited_questions_report_stale(self, tmp_path):
        leaves = [_leaf("a", "x")]
        _precompute(tmp_path, leaves, Chat())
        wider = choices.QuestionSet(version=1, questions=QUESTIONS.questions + [
            choices.Question(id="q3", prompt="New?", answers=[
                choices.Answer("q3__yes", "Yes", "anything at all")])])
        cov = choices.document_coverage(tmp_path, "doc", leaves, "m1", wider)
        assert cov["state"] == "stale"


class TestParallelPrecompute:
    """Judgements are independent, so they run on the caller's pool.

    Serially, a corpus-wide pass left three of four Ollama slots idle for its
    entire duration.
    """

    def _pool(self):
        from concurrent.futures import ThreadPoolExecutor
        return ThreadPoolExecutor(max_workers=4)

    def test_parallel_and_serial_agree(self, tmp_path):
        leaves = [_leaf(f"n{i}", f"passage {i}") for i in range(6)]
        chat = Chat({"passage 3": ["q1__icu"]})
        with self._pool() as pool:
            _precompute(tmp_path, leaves, chat, submit=pool.submit)
        parallel = choices.FacetStore(tmp_path, "doc").data

        serial_dir = tmp_path / "serial"
        serial_dir.mkdir()
        choices.precompute_document("doc", leaves, serial_dir, chat, json.loads,
                                    "m1", questions=QUESTIONS)
        assert parallel == choices.FacetStore(serial_dir, "doc").data

    def test_every_judgement_is_stored_exactly_once(self, tmp_path):
        leaves = [_leaf(f"n{i}", f"passage {i}") for i in range(8)]
        with self._pool() as pool:
            report = _precompute(tmp_path, leaves, Chat(), submit=pool.submit)
        store = choices.FacetStore(tmp_path, "doc")
        assert report["calls"] == 16
        assert len(store.data) == 8
        assert all(set(v) == {"q1", "q2"} for v in store.data.values())

    def test_cancelling_keeps_what_was_already_judged(self, tmp_path):
        """A cancelled pass is progress, not waste — the store is incremental,
        so the next run picks up where this one stopped."""
        leaves = [_leaf(f"n{i}", f"passage {i}") for i in range(10)]
        chat = Chat()
        report = choices.precompute_document(
            "doc", leaves, tmp_path, chat, json.loads, "m1", questions=QUESTIONS,
            cancelled=lambda: len(chat.calls) >= 4)

        assert report["cancelled"]
        assert 0 < report["calls"] < 20
        store = choices.FacetStore(tmp_path, "doc")
        assert 0 < len(store.data) < 10, "partial, and saved"

    def test_a_resumed_pass_only_judges_what_is_left(self, tmp_path):
        leaves = [_leaf(f"n{i}", f"passage {i}") for i in range(5)]
        chat = Chat()
        choices.precompute_document("doc", leaves, tmp_path, chat, json.loads,
                                    "m1", questions=QUESTIONS,
                                    cancelled=lambda: len(chat.calls) >= 4)
        done = choices.FacetStore(tmp_path, "doc").data

        resumed = Chat()
        report = _precompute(tmp_path, leaves, resumed)
        assert report["reused"] == sum(len(v) for v in done.values())
        assert len(choices.FacetStore(tmp_path, "doc").data) == 5

    def test_a_worker_that_explodes_is_recall_biased_like_any_other_failure(
            self, tmp_path):
        def broken(prompt):
            raise ConnectionError("instance down")

        with self._pool() as pool:
            report = choices.precompute_document(
                "doc", [_leaf("a", "x")], tmp_path, broken, json.loads, "m1",
                questions=QUESTIONS, submit=pool.submit)

        assert report["errors"]
        store = choices.FacetStore(tmp_path, "doc")
        assert set(store.get(choices.leaf_hash("A", "x"), "q1")) == {"q1__icu", "q1__ward"}


class TestStaleJudgementsAreNotEvidence:
    """Found by running the app, not by the suite.

    `FacetStore` loads whatever is on disk — correct for the precompute, which
    must see the old data to decide what to redo. At QUERY time it meant a
    corpus judged against the previous question wording still filtered live
    searches, while the status endpoint reported "stale" beside it. The two
    disagreed, and only one of them was visible to the user.
    """

    def _stale(self, tmp_path):
        leaves = [_leaf("a", "icu passage"), _leaf("b", "other")]
        _precompute(tmp_path, leaves, Chat({"icu passage": ["q1__icu"]}))
        return leaves

    def test_a_raw_store_still_reads_the_old_data(self, tmp_path):
        """The precompute depends on this, so it is pinned deliberately."""
        leaves = self._stale(tmp_path)
        raw = choices.FacetStore(tmp_path, "doc")
        assert choices.selected_leaf_ids(raw, leaves, ["q1__icu"], QUESTIONS) == {"a"}

    def test_a_model_change_stops_the_old_judgements_filtering(self, tmp_path):
        leaves = self._stale(tmp_path)
        store = choices.open_store(tmp_path, "doc", "a-different-model", QUESTIONS)
        selection = choices.select_leaves(store, leaves, ["q1__icu"], QUESTIONS)
        assert selection.kept == {"a", "b"}, "unjudged means searched, not filtered"
        assert selection.fully_unjudged

    def test_edited_questions_stop_the_old_judgements_filtering(self, tmp_path):
        leaves = self._stale(tmp_path)
        reworded = choices.QuestionSet(version=1, questions=[
            choices.Question(id="q1", prompt="Setting?", answers=[
                choices.Answer("q1__icu", "ICU", "critically ill, ventilated patients"),
                choices.Answer("q1__ward", "Ward", "general ward")]),
            QUESTIONS.questions[1]])
        store = choices.open_store(tmp_path, "doc", "m1", reworded)
        assert choices.select_leaves(store, leaves, ["q1__icu"], reworded).fully_unjudged

    def test_an_unchanged_set_is_still_trusted(self, tmp_path):
        leaves = self._stale(tmp_path)
        store = choices.open_store(tmp_path, "doc", "m1", QUESTIONS)
        assert choices.selected_leaf_ids(store, leaves, ["q1__icu"], QUESTIONS) == {"a"}


class TestInFlightWindow:
    """The pool is SHARED with live retrieval.

    Submitting a whole document at once put every question asked during a
    precompute behind hundreds of judgements. A bounded window keeps the
    instances busy without owning the queue — and it is what makes cancelling
    take effect promptly, since queued work is what there is to cancel.
    """

    class CountingPool:
        """Records how many submitted tasks are outstanding at any moment."""

        def __init__(self, workers=4):
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(max_workers=workers)
            self.outstanding = 0
            self.peak = 0
            self._lock = __import__("threading").Lock()

        def submit(self, fn, arg):
            with self._lock:
                self.outstanding += 1
                self.peak = max(self.peak, self.outstanding)

            def run():
                try:
                    return fn(arg)
                finally:
                    with self._lock:
                        self.outstanding -= 1
            return self.pool.submit(run)

        def shutdown(self):
            self.pool.shutdown()

    def test_never_queues_more_than_the_window(self, tmp_path):
        leaves = [_leaf(f"n{i}", f"passage {i}") for i in range(25)]
        pool = self.CountingPool()
        try:
            _precompute(tmp_path, leaves, Chat(), submit=pool.submit, max_inflight=6)
        finally:
            pool.shutdown()
        assert pool.peak <= 6, f"queued {pool.peak} at once, window was 6"
        assert len(choices.FacetStore(tmp_path, "doc").data) == 25, (
            "a window must not lose work")

    def test_cancelling_stops_promptly_rather_than_draining(self, tmp_path):
        """The whole point of the window: cancel used to have to wait for
        every already-queued judgement to run."""
        leaves = [_leaf(f"n{i}", f"passage {i}") for i in range(40)]
        chat = Chat()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            report = _precompute(tmp_path, leaves, chat, submit=pool.submit,
                                 max_inflight=4,
                                 cancelled=lambda: len(chat.calls) >= 8)
        assert report["cancelled"]
        assert report["calls"] < 80, "did not drain the whole document"
        assert len(chat.calls) < 40
