"""
Multiple-choice pre-filtering.

Before the user types a query, they answer a few fixed multiple-choice
questions. Each answer maps to a set of leaves that were judged relevant to it
ONCE, at index time. At query time the selected answers' leaf sets are unioned
and the user's question runs only over that subtree.

Why the work happens at index time: judging every leaf against every question
is expensive, but the questions are fixed and the documents are not changing
during a session. The result is keyed by the leaf's content hash, so it
persists across re-indexing and only leaves whose text actually changed are
recomputed.

Cost is one call per (leaf, QUESTION), not per (leaf, answer): the prompt lists
that question's answers and asks which apply, so a question with six answers
still costs one call.

Selection semantics are UNION, per the design decision: selecting more answers
widens the candidate set. The candidate count is reported back so the effect of
each selection is visible rather than surprising.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
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


# Placeholder questions. `label` is the user-facing wording and `facet` is the
# criterion the model judges passages against; they are separate so the two can
# be tuned independently.
DEFAULT_QUESTIONS = QuestionSet(version=1, questions=[
    Question(
        id="q_setting", prompt="Where is the patient being treated?",
        help_text="Placeholder — replace with real triage questions.",
        answers=[
            Answer("q_setting__icu", "Intensive care",
                   "critical care, intensive care, ventilated or unstable patients"),
            Answer("q_setting__ward", "General ward",
                   "general inpatient ward care"),
            Answer("q_setting__community", "Community / outpatient",
                   "primary care, outpatient or community management"),
        ]),
    Question(
        id="q_population", prompt="Which population?",
        help_text="Placeholder — replace with real triage questions.",
        answers=[
            Answer("q_population__adult", "Adults", "adult patients"),
            Answer("q_population__paediatric", "Children",
                   "paediatric, neonatal or adolescent patients"),
            Answer("q_population__pregnancy", "Pregnancy",
                   "pregnancy, obstetric or peripartum care"),
        ]),
    Question(
        id="q_intent", prompt="What are you trying to decide?",
        help_text="Placeholder — replace with real triage questions.",
        answers=[
            Answer("q_intent__diagnosis", "Reach a diagnosis",
                   "diagnosis, investigation, testing or assessment"),
            Answer("q_intent__treatment", "Choose a treatment",
                   "treatment, therapy, drug choice or dosing"),
            Answer("q_intent__monitoring", "Monitor or follow up",
                   "monitoring, follow-up, complications or safety"),
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


def _cache_key(questions: QuestionSet) -> str:
    """Everything a stored judgement depends on."""
    return (f"v{questions.version}"
            f"|p{app_config.PROMPT_VERSIONS['facet_leaf']}")


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
) -> dict:
    """Judge every leaf against every question, reusing stored results.

    `chat(prompt) -> str` and `parse_json(raw) -> object` are injected so this
    module stays independent of the Ollama plumbing (and trivially testable).

    Returns {"calls": n, "reused": n, "leaves": n, "errors": [...]}.
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

    calls = reused = 0
    errors: list[str] = []

    for i, leaf in enumerate(leaves):
        h = leaf_hash(leaf.title, leaf.content)
        for question in questions.questions:
            if store.get(h, question.id) is not None:
                reused += 1
                continue

            # Only THIS question's answers are valid here. Validating against
            # the whole set would let an answer from another question be filed
            # under this one.
            valid_ids = {a.id for a in question.answers}
            answers_block = "\n".join(
                f"- id={a.id} | {a.facet}" for a in question.answers)
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
                picked = []
                for a in raw_ids:
                    resolved = normalise_answer_id(a, valid_ids)
                    if resolved is not None:
                        if resolved not in picked:
                            picked.append(resolved)
                    else:
                        print(f"[choices] ignoring unrecognised answer {a!r} "
                              f"for {question.id}", file=sys.stderr)
                store.put(h, question.id, picked)
            except Exception as exc:
                # Recall-biased on failure too: an unjudged leaf is treated as
                # relevant to every answer rather than being excluded from all
                # of them, since exclusion here is unrecoverable.
                store.put(h, question.id, [a.id for a in question.answers])
                errors.append(f"{leaf.node_id}/{question.id}: {exc}")
            calls += 1

        if progress:
            progress({"doc": doc_id, "done": i + 1, "total": len(leaves)})

    store.save(model, key)
    return {"calls": calls, "reused": reused, "leaves": len(leaves), "errors": errors}


# ---------------------------------------------------------------------------
# Query-time selection
# ---------------------------------------------------------------------------

def selected_leaf_ids(
    store: FacetStore,
    leaves: list,
    selected_answers: list[str],
) -> set[str]:
    """Leaf ids matching ANY selected answer (union).

    More selections means a WIDER candidate set, by design — each answer
    contributes its own leaves rather than constraining the others.
    """
    if not selected_answers:
        return {leaf.node_id for leaf in leaves}

    wanted = set(selected_answers)
    out: set[str] = set()
    for leaf in leaves:
        h = leaf_hash(leaf.title, leaf.content)
        judgements = store.data.get(h) or {}
        for answer_ids in judgements.values():
            if wanted & set(answer_ids or []):
                out.add(leaf.node_id)
                break
    return out


def answers_for_leaf(store: FacetStore, leaf, selected_answers: list[str]) -> list[str]:
    """Which of the selected answers put this leaf in the candidate set.

    Under union semantics leaves genuinely differ in membership, so this is
    the informative part: it says WHY a passage is in play.
    """
    h = leaf_hash(leaf.title, leaf.content)
    judgements = store.data.get(h) or {}
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
