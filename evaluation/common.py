"""
Shared pieces of the evaluation: the query file format, the corpus as a flat
list of leaves, and the per-query record derived from one /api/chat payload.

Everything that interprets a run lives in `derive_record`, so the driver, the
report and the tests all read a payload the same way.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pageindex  # noqa: E402
import ranking  # noqa: E402

# ---------------------------------------------------------------------------
# Query file
#
#   id,set,pair_id,doc,node_id,question,edit_term
#
#   id        unique, e.g. A017 / B017
#   set       A (gold query) | B (paired edit)
#   pair_id   for a B row, the id of its A partner; blank for A rows
#   doc       index stem (index/<doc>.json)
#   node_id   the gold leaf's nodeId within that document
#   question  the query text
#   edit_term optional; for B rows, the invented detail that was inserted
#             (e.g. "Pinocchio Nasal Elongation Disorder"), so the report can
#             check whether STILL_NEEDED names it
#
# The user's earlier CSV (document,page,question) is page-based and predates
# the hand-written markdown corpus; there is no page to map. Gold is a chunk.
# ---------------------------------------------------------------------------

QUERY_COLUMNS = ["id", "set", "pair_id", "doc", "node_id", "question", "edit_term"]


@dataclass
class Query:
    id: str
    set: str
    pair_id: str
    doc: str
    node_id: str
    question: str
    edit_term: str = ""

    @property
    def gold(self) -> tuple[str, str]:
        return (self.doc, self.node_id)


def load_queries(path: Path, sets: Optional[Iterable[str]] = None) -> list[Query]:
    wanted = {s.strip().upper() for s in sets} if sets else None
    out: list[Query] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in QUERY_COLUMNS[:6] if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; expected {QUERY_COLUMNS}")
        for row in reader:
            q = Query(
                id=(row.get("id") or "").strip(),
                set=(row.get("set") or "").strip().upper(),
                pair_id=(row.get("pair_id") or "").strip(),
                doc=(row.get("doc") or "").strip(),
                node_id=(row.get("node_id") or "").strip(),
                question=(row.get("question") or "").strip(),
                edit_term=(row.get("edit_term") or "").strip(),
            )
            if not q.id and not q.question:
                continue  # blank line
            if wanted and q.set not in wanted:
                continue
            out.append(q)
    return out


def write_queries(path: Path, queries: list[Query]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=QUERY_COLUMNS)
        writer.writeheader()
        for q in queries:
            writer.writerow(asdict(q))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

@dataclass
class Leaf:
    doc: str
    node_id: str
    title: str
    breadcrumb: str
    summary: str
    content: str
    order: int  # position in (sorted doc, tree order) — the system's own order

    @property
    def key(self) -> tuple[str, str]:
        return (self.doc, self.node_id)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc}/{self.node_id}"


def load_corpus_leaves(index_dir: Optional[Path] = None) -> list[Leaf]:
    """Every leaf in the index, in exactly the order the server walks them:
    index files sorted by name, then tree order within each."""
    index_dir = Path(index_dir or pageindex.INDEX_DIR)
    leaves: list[Leaf] = []
    for path in sorted(index_dir.glob("*.json")):
        try:
            nodes = [pageindex._node_from_dict(d)
                     for d in pageindex.read_index_file(path)["nodes"]]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            print(f"[corpus] skipping unreadable index {path.name}: {exc}", file=sys.stderr)
            continue
        by_id = pageindex._build_nodes_by_id(nodes)
        parents = pageindex._build_parent_map(nodes)
        for leaf in pageindex._collect_leaves(nodes):
            leaves.append(Leaf(
                doc=path.stem, node_id=leaf.node_id, title=leaf.title,
                breadcrumb=pageindex._make_breadcrumb(leaf.node_id, parents, by_id),
                summary=leaf.summary or "", content=leaf.content or "",
                order=len(leaves)))
    return leaves


def leaf_text_for_ranking(leaf: Leaf) -> str:
    """The exact text pageindex.evaluate_ranked scores — so an offline BM25
    ranking is the same function the system runs, over a different candidate
    set (the whole corpus rather than the post-pruning survivors)."""
    return f"{leaf.title}\n{leaf.summary}\n{leaf.content}"


def rank_corpus(leaves: list[Leaf], query: str) -> list[ranking.Scored]:
    return ranking.rank(leaves, query, text_of=leaf_text_for_ranking)


# ---------------------------------------------------------------------------
# Per-query record, derived from one /api/chat payload
# ---------------------------------------------------------------------------

# How a gold leaf can end a run. `deferred_judged_relevant` is the one node
# the evaluator read and accepted before the budget cut it — a positive
# verdict, not a censored one — and is kept apart from the unevaluated tail.
GOLD_OUTCOMES = ("retrieved", "rejected", "pruned", "deferred",
                 "deferred_judged_relevant", "error", "untouched")

# Outcomes that count as "the retriever never formed a verdict": excluded
# from the retrieved/missed split and reported on their own.
CENSORED = {"deferred"}


@dataclass
class Record:
    id: str
    set: str
    pair_id: str
    query: str
    gold_doc: str
    gold_node_id: str
    edit_term: str
    run_id: str
    # retrieval
    n_retrieved: int
    retrieved: list            # [[doc, node_id], ...] in source order
    gold_retrieved: bool
    gold_outcome: str          # one of GOLD_OUTCOMES
    gold_bm25_rank: Optional[int]   # 0-based rank among post-pruning survivors, or None
    gold_bm25_score: Optional[float]
    # answer
    grounding_status: str
    cited: list                # [[doc, node_id], ...]
    gold_cited: bool
    judgment_raw: Optional[str]
    judgment_label: str        # sufficient | insufficient | unparsed | missing
    judgment_ui_reading: str   # what the product's prefix match would say
    still_needed: list
    names_edit: Optional[bool]  # B rows only: does STILL_NEEDED mention edit_term
    cell: str
    # cost
    timing: dict
    budget: dict
    index_fingerprint: str = ""
    error: str = ""
    answer_text: str = ""
    extra: dict = field(default_factory=dict)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


def mentions(haystack: str, needle: str) -> bool:
    """Loose match: at least half the needle's content words appear."""
    n = _tokens(needle)
    if not n:
        return False
    return len(n & _tokens(haystack)) * 2 >= len(n)


def classify_cell(gold_outcome: str, gold_retrieved: bool, label: str) -> str:
    """The four cells, conditional on retrieval, with the censored and the
    unlabeled cases kept out of them rather than folded in."""
    if gold_outcome in CENSORED:
        side = "deferred"
    elif gold_retrieved:
        side = "retrieved"
    else:
        side = "missed"
    if label not in ("sufficient", "insufficient"):
        return f"{side}_unlabeled"
    return f"{side}_{label}"


def derive_record(q: Query, payload: dict, index_fingerprint: str = "") -> Record:
    """Read one /api/chat response into a flat record."""
    if "error" in payload and "grounding" not in payload:
        return Record(
            id=q.id, set=q.set, pair_id=q.pair_id, query=q.question,
            gold_doc=q.doc, gold_node_id=q.node_id, edit_term=q.edit_term,
            run_id="", n_retrieved=0, retrieved=[], gold_retrieved=False,
            gold_outcome="untouched", gold_bm25_rank=None, gold_bm25_score=None,
            grounding_status="", cited=[], gold_cited=False, judgment_raw=None,
            judgment_label="missing", judgment_ui_reading="sufficient",
            still_needed=[], names_edit=None, cell="error", timing={}, budget={},
            index_fingerprint=index_fingerprint, error=str(payload.get("error")))

    sources = (payload.get("grounding") or {}).get("sources") or []
    retrieved = [[s["doc"], s["node_id"]] for s in sources]
    gold_retrieved = [q.doc, q.node_id] in retrieved

    results = ((payload.get("run") or {}).get("results")) or payload.get("results") or {}
    meta = ((results.get(q.doc) or {}).get("node_meta") or {}).get(q.node_id) or {}
    status = meta.get("status") or "untouched"
    if status == "deferred" and meta.get("judged_relevant"):
        status = "deferred_judged_relevant"
    if status not in GOLD_OUTCOMES:
        status = "untouched"
    if gold_retrieved and status != "retrieved":
        # Defensive: the sources list is what the answer was built from.
        status = "retrieved"

    answer = payload.get("answer") or {}
    answer_text = answer.get("content") or ""
    judgment = payload.get("judgment")
    if judgment is None:
        # An older server without the judgment field: interpret locally so
        # the record is still complete, using the same function.
        import synthesis
        judgment = synthesis.interpret_judgment(
            {k: v for k, v in answer.items() if k != "content"})
    grounding = (payload.get("grounding") or {}).get("status") or ""

    import synthesis
    _, valid_cited = synthesis.grounding_status(answer_text, len(sources))
    by_n = {s["n"]: s for s in sources}
    cited = [[by_n[n]["doc"], by_n[n]["node_id"]] for n in sorted(valid_cited) if n in by_n]
    gold_cited = [q.doc, q.node_id] in cited

    still_needed = list(judgment.get("still_needed") or [])
    names_edit = None
    if q.set == "B" and q.edit_term:
        names_edit = mentions(" ".join(still_needed), q.edit_term)

    return Record(
        id=q.id, set=q.set, pair_id=q.pair_id, query=q.question,
        gold_doc=q.doc, gold_node_id=q.node_id, edit_term=q.edit_term,
        run_id=str(payload.get("run_id") or ""),
        n_retrieved=len(sources), retrieved=retrieved,
        gold_retrieved=gold_retrieved, gold_outcome=status,
        gold_bm25_rank=meta.get("bm25_rank"), gold_bm25_score=meta.get("bm25_score"),
        grounding_status=grounding, cited=cited, gold_cited=gold_cited,
        judgment_raw=judgment.get("raw"), judgment_label=judgment.get("label") or "missing",
        judgment_ui_reading=judgment.get("ui_reading") or "sufficient",
        still_needed=still_needed, names_edit=names_edit,
        cell=classify_cell(status, gold_retrieved, judgment.get("label") or "missing"),
        timing=dict(payload.get("timing") or {}),
        budget={k: v for k, v in (payload.get("budget") or {}).items()
                if k != "deferred_nodes"},
        index_fingerprint=index_fingerprint,
        answer_text=answer_text,
    )


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def load_records(run_dir: Path) -> list[dict]:
    return read_jsonl(Path(run_dir) / "records.jsonl")


def percentile(values: list, p: float) -> Optional[float]:
    """Nearest-rank percentile; None on an empty list."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    import math
    k = max(0, min(len(vals) - 1, math.ceil(p / 100.0 * len(vals)) - 1))
    return float(vals[k])


def pair_drift(a: dict, b: dict) -> dict:
    """Did the paired edit move retrieval? Compares the two retrieved sets."""
    ra = {tuple(x) for x in a.get("retrieved") or []}
    rb = {tuple(x) for x in b.get("retrieved") or []}
    union = ra | rb
    return {
        "same_set": ra == rb,
        "jaccard": (len(ra & rb) / len(union)) if union else 1.0,
        "only_a": sorted(ra - rb),
        "only_b": sorted(rb - ra),
        "gold_in_both": bool(a.get("gold_retrieved") and b.get("gold_retrieved")),
    }
