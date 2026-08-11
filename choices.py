"""
Multiple-choice pre-filtering.

Before the user types a query, they answer a few fixed multiple-choice
questions. Each answer maps to a set of leaves that were judged relevant to it
ONCE, at index time. At query time the selected answers pick a leaf set and the
user's question runs only over that subtree — no model calls, just lookups.

Why the work happens at index time: judging every leaf against every question
is expensive, but the questions are fixed and the documents are not changing
during a session. The result is keyed by the leaf's content hash, so it
persists across re-indexing and only leaves whose text actually changed are
recomputed.

Cost is one call per (leaf, QUESTION), not per (leaf, answer): the prompt lists
that question's answers and asks which apply, so a question with six answers
still costs one call.

Selection semantics are a deliberate, user-visible choice:

  "any"  (default) — a leaf is a candidate if it matches ANY selected answer.
                     Selecting more answers WIDENS the search. Recall-first:
                     in a clinical setting a passage wrongly excluded here can
                     never reach the user, and that is the expensive error.

  "all"  (Fast mode, opt-in) — OR within a question, AND across questions. A
                     leaf must satisfy every question the user answered.
                     Much smaller candidate set and a much faster query, at a
                     real cost in recall — hence opt-in, never the default.

The resulting candidate count is reported back for both, so the effect of
every click is visible rather than inferred.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config as app_config

ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "config" / "choice_questions.json"
FACETS_DIRNAME = ".facets"


# The facet prompt is deliberately RECALL-BIASED. A false negative here is
# unrecoverable — a leaf excluded at this stage can never reach the user's
# query, no matter how well it would have answered it. A false positive only
# costs a little of the Phase-3 budget.
FACET_LEAF_PROMPT = """\
Below is a passage from a document, and a question with several possible
answers. Decide which of those answers the passage is relevant to.

Question: QUESTION_PLACEHOLDER

Answers:
ANSWERS_PLACEHOLDER

--- PASSAGE ---
CONTENT_PLACEHOLDER
--- END ---

Rules:
- Select every answer the passage could be relevant to. Include it if there is
  any reasonable chance — a passage wrongly excluded here can never be shown to
  the user, while a passage wrongly included costs almost nothing.
- A passage can be relevant to several answers, or to none.
- Use the ids exactly as written. Never invent an id.

Output exactly this JSON object and nothing else:
  {"answers": ["<id>", ...]}
"""


@dataclass
class Answer:
    id: str
    label: str            # shown to the user
    facet: str            # what the model judges passages against


@dataclass
class Question:
    id: str
    prompt: str
    answers: list[Answer]
    multi_select: bool = True
    help_text: str = ""


@dataclass
class QuestionSet:
    version: int = 1
    questions: list[Question] = field(default_factory=list)

    def answer_ids(self) -> set[str]:
        return {a.id for q in self.questions for a in q.answers}

    def question_of(self, answer_id: str) -> Optional[Question]:
        for q in self.questions:
            if any(a.id == answer_id for a in q.answers):
                return q
        return None

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "questions": [
                {"id": q.id, "prompt": q.prompt, "multi_select": q.multi_select,
                 "help_text": q.help_text,
                 "answers": [{"id": a.id, "label": a.label, "facet": a.facet}
                             for a in q.answers]}
                for q in self.questions
            ],
        }


# The fallback question set, used only when `config/choice_questions.json`
# cannot be read. That file is authoritative and this is deliberately a subset
# of it — a corrupt config should degrade to a usable filter, not to nothing.
#
# `label` is the user-facing wording and `facet` is the criterion the model
# judges passages against; they are separate so the two can be tuned
# independently. Facets are written to DISCRIMINATE: one that could describe
# half the library makes its answer useless as a filter.
DEFAULT_QUESTIONS = QuestionSet(version=2, questions=[
    Question(
        id="q_domain", prompt="What is the question about?",
        help_text="The clinical area. Pick more than one if the question spans them.",
        answers=[
            Answer("q_domain__infection", "Infection & antimicrobials",
                   "infection, antibiotics, antimicrobial choice or duration, "
                   "cultures, sepsis, resistant organisms and isolation"),
            Answer("q_domain__cardiac", "Cardiac & vascular",
                   "chest pain, acute coronary syndrome, ECG, cardiac biomarkers, "
                   "blood pressure and cardiovascular risk"),
            Answer("q_domain__metabolic", "Metabolic & endocrine",
                   "diabetes, glycaemic targets, insulin and metabolic complications"),
            Answer("q_domain__sedation", "Sedation, analgesia & delirium",
                   "sedation, analgesia, delirium and neuromuscular blockade"),
        ]),
    Question(
        id="q_setting", prompt="Where is the patient?",
        help_text="Where care is delivered — this changes the regimen more than anything else.",
        answers=[
            Answer("q_setting__icu", "Intensive care",
                   "critical care: ventilated, unstable or organ-supported patients"),
            Answer("q_setting__ward", "Hospital ward",
                   "inpatient care outside intensive care: admission criteria, "
                   "inpatient regimens and discharge planning"),
            Answer("q_setting__community", "Outpatient / community",
                   "management outside hospital: primary care, oral outpatient "
                   "regimens and routine follow-up"),
            Answer("q_setting__periop", "Theatre / periprocedural",
                   "operating theatre, anaesthesia, surgical prophylaxis and "
                   "peri-operative management"),
        ]),
    Question(
        id="q_stage", prompt="What do you need to decide?",
        help_text="Where you are in the decision — the same topic reads differently at each stage.",
        answers=[
            Answer("q_stage__diagnose", "Diagnose or assess",
                   "presentation, investigations, diagnostic criteria, severity "
                   "scores and risk stratification"),
            Answer("q_stage__treat", "Choose treatment",
                   "drug selection, dose, route, regimen, escalation and procedures"),
            Answer("q_stage__monitor", "Monitor or follow up",
                   "monitoring parameters and targets, adverse effects, "
                   "complications, safety and follow-up"),
        ]),
])


_lock = threading.Lock()
_cached: Optional[QuestionSet] = None


def load_questions(path: Optional[Path] = None) -> QuestionSet:
    """Load the question set, falling back to the placeholders."""
    global _cached
    p = path or QUESTIONS_PATH
    with _lock:
        if path is None and _cached is not None:
            return _cached
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            qs = QuestionSet(
                version=int(data.get("version", 1)),
                questions=[
                    Question(
                        id=q["id"], prompt=q["prompt"],
                        multi_select=bool(q.get("multi_select", True)),
                        help_text=q.get("help_text", ""),
                        answers=[Answer(a["id"], a["label"], a.get("facet", a["label"]))
                                 for a in q.get("answers", [])],
                    )
                    for q in data.get("questions", [])
                ])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            if p.exists():
                print(f"[choices] could not read {p}: {exc} — using placeholders",
                      file=sys.stderr)
            qs = DEFAULT_QUESTIONS
        if path is None:
            _cached = qs
        return qs


def write_default_questions(path: Optional[Path] = None) -> Path:
    p = path or QUESTIONS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(DEFAULT_QUESTIONS.to_dict(), indent=2), encoding="utf-8")
    return p


def reset_cache() -> None:
    global _cached
    with _lock:
        _cached = None


# ---------------------------------------------------------------------------
# The persisted per-leaf judgements
# ---------------------------------------------------------------------------

def normalise_answer_id(raw, valid_ids: set[str]) -> Optional[str]:
    """Recover an answer id from whatever shape the model echoed it back in.

    The prompt lists answers as "- id=<id> | <facet>", and the model routinely
    replies with the decoration attached — "id=q_population__adult", or the
    whole line. Matching the raw string against the known ids silently dropped
    every one of those, so leaves were stored as matching NOTHING and the
    pre-filter judged almost the entire library irrelevant to every answer.

    Returns None only when nothing in the string corresponds to a real id.
    """
    text = str(raw).strip().lstrip("-*• ").strip().strip('"\'')
    candidates = [text]
    if "|" in text:
        candidates += [part.strip() for part in text.split("|")]
    expanded = []
    for c in candidates:
        expanded.append(c)
        if c.startswith("id="):
            expanded.append(c[3:].strip())
    for c in expanded:
        if c in valid_ids:
            return c
    # Last resort: an id appearing anywhere in the string.
    for known in valid_ids:
        if known in text:
            return known
    return None


def leaf_hash(title: str, content: Optional[str]) -> str:
    return hashlib.sha256(f"{title}\n{content or ''}".encode("utf-8")).hexdigest()


def questions_fingerprint(questions: QuestionSet) -> str:
    """A hash of the question set's actual content.

    `version` is a hand-maintained integer and hand-maintained integers get
    forgotten: edit a facet's wording, add an answer, and every stored
    judgement is now answering a question that no longer exists — silently,
    because the version still matches. Hashing the content means an edit
    invalidates whether or not anyone remembered to bump anything.
    """
    payload = json.dumps(questions.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _cache_key(questions: QuestionSet) -> str:
    """Everything a stored judgement depends on."""
    return (f"v{questions.version}"
            f"|p{app_config.PROMPT_VERSIONS['facet_leaf']}"
            f"|q{questions_fingerprint(questions)}")


class FacetStore:
    """`index/.facets/<doc_id>.json` — {leaf_hash: {question_id: [answer_ids]}}"""

    def __init__(self, index_dir: Path, doc_id: str):
        self.dir = Path(index_dir) / FACETS_DIRNAME
        self.path = self.dir / f"{doc_id}.json"
        self.data: dict = {}
        self.meta: dict = {}
        self.load()

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.meta = raw.get("meta", {})
            self.data = raw.get("leaves", {})
        except (OSError, json.JSONDecodeError):
            self.meta, self.data = {}, {}

    def save(self, model: str, key: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "meta": {"model": model, "key": key},
            "leaves": self.data,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    def is_valid_for(self, model: str, key: str) -> bool:
        return self.meta.get("model") == model and self.meta.get("key") == key

    def invalidate(self) -> None:
        self.data, self.meta = {}, {}

    def get(self, leaf_hash_value: str, question_id: str) -> Optional[list[str]]:
        entry = self.data.get(leaf_hash_value)
        if entry is None:
            return None
        return entry.get(question_id)

    def put(self, leaf_hash_value: str, question_id: str, answer_ids: list[str]) -> None:
        self.data.setdefault(leaf_hash_value, {})[question_id] = answer_ids

    def prune_to(self, live_hashes: set[str]) -> None:
        """Drop judgements for leaves that no longer exist."""
        for stale in set(self.data) - live_hashes:
            self.data.pop(stale, None)


def open_store(index_dir: Path, doc_id: str, model: str,
               questions: Optional[QuestionSet] = None) -> FacetStore:
    """A store whose contents can be TRUSTED at query time.

    `FacetStore` loads whatever is on disk; that is right for the precompute,
    which needs to see the old data to decide what to redo. It is wrong for
    retrieval: judgements made against different question wording, a different
    prompt or a different model are not evidence about the questions being
    asked now. Reading them anyway filtered live queries on stale judgements
    while the status endpoint reported, correctly, "stale" — the two disagreed
    and only one of them was visible.

    An emptied store makes every leaf unjudged, so the document is searched
    whole and reported as unfiltered. Recall-first, and it says so.
    """
    store = FacetStore(index_dir, doc_id)
    if store.data and not store.is_valid_for(
            model, _cache_key(questions if questions is not None else load_questions())):
        store.invalidate()
    return store


def document_coverage(index_dir: Path, doc_id: str, leaves: list, model: str,
                      questions: QuestionSet) -> dict:
    """Whether this document's judgements are usable, read from disk.

    The precompute's own progress lives in a process variable, so a server
    restart used to report "not computed yet" over a fully populated store —
    and never noticed a document ingested after the last run. Disk is the only
    thing that actually knows.

    state: "ready" | "partial" | "stale" | "missing"
    """
    store = FacetStore(index_dir, doc_id)
    key = _cache_key(questions)
    if not store.path.exists():
        state = "missing"
    elif not store.is_valid_for(model, key):
        # Judged against a different model, prompt or question set.
        state = "stale"
    else:
        judged = sum(1 for leaf in leaves
                     if store.data.get(leaf_hash(leaf.title, leaf.content)))
        state = "ready" if judged >= len(leaves) else "partial"
        return {"doc": doc_id, "state": state, "leaves": len(leaves),
                "judged": judged}
    return {"doc": doc_id, "state": state, "leaves": len(leaves), "judged": 0}


# ---------------------------------------------------------------------------
# The precompute pass
# ---------------------------------------------------------------------------

def precompute_document(
    doc_id: str,
    leaves: list,
    index_dir: Path,
    chat,
    parse_json,
    model: str,
    questions: Optional[QuestionSet] = None,
    progress=None,
    submit=None,
    cancelled=None,
    max_inflight: int = 8,
) -> dict:
    """Judge every leaf against every question, reusing stored results.

    `chat(prompt) -> str` and `parse_json(raw) -> object` are injected so this
    module stays independent of the Ollama plumbing (and trivially testable).

    `submit(fn, arg) -> Future` runs the judgements in parallel on the caller's
    pool, at most `max_inflight` at a time so a shared pool stays responsive to
    live queries; omit it for the serial path. `cancelled() -> bool` is polled
    between judgements — whatever finished is still saved, since the store is
    incremental and a partial pass is never wasted work.

    Returns {"calls", "reused", "leaves", "pending", "cancelled", "errors"}.
    """
    questions = questions or load_questions()
    store = FacetStore(index_dir, doc_id)
    key = _cache_key(questions)
    if not store.is_valid_for(model, key):
        # The question set, the prompt or the model changed — nothing stored
        # under the old key can be trusted.
        store.invalidate()

    live_hashes = {leaf_hash(l.title, l.content) for l in leaves}
    store.prune_to(live_hashes)

    reused = 0
    pending: list[tuple] = []
    for leaf in leaves:
        h = leaf_hash(leaf.title, leaf.content)
        for question in questions.questions:
            if store.get(h, question.id) is not None:
                reused += 1
            else:
                pending.append((leaf, h, question))

    errors: list[str] = []
    done = 0

    def report():
        if progress:
            progress({"doc": doc_id, "done": done, "total": len(pending),
                      "leaves": len(leaves)})

    def judge(job):
        leaf, _h, question = job
        return _judge_leaf(leaf, question, chat, parse_json)

    def record(job, picked, error):
        nonlocal done
        leaf, h, question = job
        store.put(h, question.id, picked)
        if error:
            errors.append(f"{leaf.node_id}/{question.id}: {error}")
        done += 1
        report()

    report()
    if submit is None:
        # Serial: the deterministic path, and what the tests exercise.
        for job in pending:
            if cancelled and cancelled():
                break
            record(job, *judge(job))
    else:
        # Parallel through the caller's pool — the judgements are independent,
        # and running them one at a time left three of four Ollama slots idle
        # for the whole precompute. The store is written HERE, on this thread
        # only, so the workers stay free of shared state.
        #
        # A BOUNDED window, not the whole document at once. The pool is shared
        # with live retrieval: dumping 268 judgements onto it would put every
        # question asked during a precompute behind all of them. A window keeps
        # the instances busy while leaving the queue short enough to interleave
        # — and it is what makes cancelling take effect promptly, since queued
        # work is what there is to cancel.
        futures: dict = {}
        queue = iter(pending)
        stop = False

        def fill():
            while len(futures) < max(1, max_inflight):
                job = next(queue, None)
                if job is None:
                    return
                futures[submit(judge, job)] = job

        try:
            fill()
            while futures and not stop:
                for future in list(futures):
                    if not future.done():
                        continue
                    job = futures.pop(future)
                    try:
                        picked, error = future.result()
                    except Exception as exc:   # a pool failure, not a judgement
                        picked, error = [a.id for a in job[2].answers], exc
                    record(job, picked, error)
                if cancelled and cancelled():
                    stop = True
                    break
                fill()
                if futures and not any(f.done() for f in futures):
                    time.sleep(0.02)
        finally:
            for future in futures:
                future.cancel()

    store.save(model, key)
    return {"calls": done, "reused": reused, "leaves": len(leaves),
            "pending": len(pending), "cancelled": bool(cancelled and cancelled()),
            "errors": errors}


def _judge_leaf(leaf, question: Question, chat, parse_json) -> tuple[list[str], object]:
    """One (leaf, question) judgement. Returns (answer_ids, error_or_None).

    Pure with respect to the store: it returns what it found and lets the
    caller do the writing, so this can run on a worker thread.
    """
    # Only THIS question's answers are valid here. Validating against the whole
    # set would let an answer from another question be filed under this one.
    valid_ids = {a.id for a in question.answers}
    answers_block = "\n".join(f"- id={a.id} | {a.facet}" for a in question.answers)
    prompt = (
        FACET_LEAF_PROMPT
        .replace("QUESTION_PLACEHOLDER", question.prompt)
        .replace("ANSWERS_PLACEHOLDER", answers_block)
        .replace("CONTENT_PLACEHOLDER",
                 f"{leaf.title}\n{leaf.content or '(no content)'}")
    )
    try:
        result = parse_json(chat(prompt))
        raw_ids = result.get("answers", []) if isinstance(result, dict) else result
        if not isinstance(raw_ids, list):
            raise ValueError(f"expected a list, got {type(raw_ids).__name__}")
        picked: list[str] = []
        for a in raw_ids:
            resolved = normalise_answer_id(a, valid_ids)
            if resolved is None:
                print(f"[choices] ignoring unrecognised answer {a!r} "
                      f"for {question.id}", file=sys.stderr)
            elif resolved not in picked:
                picked.append(resolved)
        return picked, None
    except Exception as exc:
        # Recall-biased on failure too: an unjudged leaf is treated as relevant
        # to every answer rather than being excluded from all of them, since
        # exclusion here is unrecoverable.
        return [a.id for a in question.answers], exc


# ---------------------------------------------------------------------------
# Query-time selection
# ---------------------------------------------------------------------------

ANY = "any"   # union across everything — the recall-first default
ALL = "all"   # OR within a question, AND across questions — Fast mode
MODES = (ANY, ALL)


def normalise_mode(mode: Optional[str]) -> str:
    """Anything unrecognised means the recall-first default.

    A typo or a stale frontend must never silently switch a clinical search
    into the narrower mode.
    """
    return mode if mode in MODES else ANY


@dataclass
class Selection:
    """The outcome of applying one set of answers to one document.

    `unjudged` is separated from `kept` deliberately. A leaf with no stored
    judgement is kept — exclusion here is unrecoverable — but it is kept for a
    completely different reason than a leaf that was judged and matched, and
    conflating the two is what let a whole unjudged document look filtered.
    """
    kept: set[str] = field(default_factory=set)
    unjudged: set[str] = field(default_factory=set)
    total: int = 0
    mode: str = ANY

    @property
    def judged(self) -> int:
        return self.total - len(self.unjudged)

    @property
    def fully_unjudged(self) -> bool:
        """No leaf in this document has any judgement at all."""
        return self.total > 0 and len(self.unjudged) == self.total


def group_by_question(
    selected_answers: list[str],
    questions: Optional[QuestionSet] = None,
) -> dict[str, set[str]]:
    """{question_id: selected answers belonging to it}.

    An answer that belongs to no known question is grouped under its own id,
    which matches no question and therefore satisfies nothing — an id from a
    stale frontend must exclude everything, not quietly widen the search.
    """
    questions = questions if questions is not None else load_questions()
    grouped: dict[str, set[str]] = {}
    for answer_id in selected_answers:
        question = questions.question_of(answer_id)
        grouped.setdefault(question.id if question else answer_id, set()).add(answer_id)
    return grouped


def select_leaves(
    store: FacetStore,
    leaves: list,
    selected_answers: list[str],
    questions: Optional[QuestionSet] = None,
    mode: str = ANY,
) -> Selection:
    """Apply the selected answers to one document's leaves.

    ANY (default): a leaf matching ANY selected answer is a candidate. More
    selections means a wider search. Recall-first, because a passage excluded
    here can never reach the user no matter how well it would have answered.

    ALL (Fast mode): OR within a question, AND across questions — "Intensive
    care" plus "Adults" means intensive care AND adults. Far fewer candidates
    and a much faster query, at a real cost in recall. On this corpus `adult`
    alone covers 54 of 67 leaves, so ANY with two answers searches nearly
    everything while ALL cuts hard — the difference is exactly why the user
    gets to choose rather than being given one of them silently.
    """
    mode = normalise_mode(mode)
    result = Selection(total=len(leaves), mode=mode)
    if not selected_answers:
        result.kept = {leaf.node_id for leaf in leaves}
        return result

    questions = questions if questions is not None else load_questions()
    known_question_ids = {q.id for q in questions.questions}
    grouped = group_by_question(selected_answers, questions)
    wanted_any = set(selected_answers)

    for leaf in leaves:
        judgements = store.data.get(leaf_hash(leaf.title, leaf.content))
        if not judgements:
            # Never judged — keep it and say so, rather than excluding a
            # passage on the strength of a judgement that was never made.
            result.unjudged.add(leaf.node_id)
            result.kept.add(leaf.node_id)
            continue

        if mode == ANY:
            if any(wanted_any & set(ids or []) for ids in judgements.values()):
                result.kept.add(leaf.node_id)
            continue

        satisfied = True
        for question_id, wanted in grouped.items():
            stored = judgements.get(question_id)
            if stored is None:
                # This leaf predates the question. Recall-biased even here: an
                # unasked question cannot be evidence of irrelevance. Unless
                # the question does not exist at all, in which case nothing
                # can ever satisfy it.
                if question_id not in known_question_ids:
                    satisfied = False
                    break
                continue
            if not wanted & set(stored):
                satisfied = False
                break
        if satisfied:
            result.kept.add(leaf.node_id)
    return result


def selected_leaf_ids(
    store: FacetStore,
    leaves: list,
    selected_answers: list[str],
    questions: Optional[QuestionSet] = None,
    mode: str = ANY,
) -> set[str]:
    """The leaf ids `select_leaves` keeps. See it for the semantics."""
    return select_leaves(store, leaves, selected_answers, questions, mode).kept


def answers_for_leaf(store: FacetStore, leaf, selected_answers: list[str]) -> list[str]:
    """Which of the selected answers this leaf was actually judged relevant to.

    Under ANY this is the informative part — leaves genuinely differ in why
    they are in play. Under ALL it mostly confirms, but it stays honest: an
    answer missing here is a question the leaf was never judged against, which
    is exactly what a user wants to see before trusting the filter.
    """
    judgements = store.data.get(leaf_hash(leaf.title, leaf.content)) or {}
    hit = {a for ids in judgements.values() for a in (ids or [])}
    return sorted(hit & set(selected_answers)) if selected_answers else sorted(hit)


def ancestor_closure(nodes: list, keep_leaf_ids: set[str]) -> set[str]:
    """The selected leaves plus every ancestor of theirs.

    Deriving the internal nodes from the leaves — rather than storing and
    combining internal-node sets directly — is what guarantees the result is a
    connected tree rather than a fragment with holes in it.
    """
    keep: set[str] = set()

    def walk(node) -> bool:
        hit = node.node_id in keep_leaf_ids
        for child in node.children:
            if walk(child):
                hit = True
        if hit:
            keep.add(node.node_id)
        return hit

    for root in nodes:
        walk(root)
    return keep


def prune_tree_to(nodes: list, keep_ids: set[str]) -> list:
    """A copy of `nodes` containing only `keep_ids`, structure preserved."""
    import copy

    def build(node):
        if node.node_id not in keep_ids:
            return None
        clone = copy.copy(node)
        clone.children = [c for c in (build(ch) for ch in node.children) if c]
        return clone

    return [n for n in (build(root) for root in nodes) if n]
