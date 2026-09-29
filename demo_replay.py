"""
Recorded-run replay for the Demo tab.

Serves the committed 100-question evaluation (evaluation/results/asepsis100/
set{A,B,C}.json + their logs) back to the UI, joined to the index so every
retrieved node id comes with its title, summary, page and an excerpt.

Nothing here calls a model or touches the retrieval path: it reads files that
were already produced. The UI labels everything it shows from here as a
recorded run.

What the records do and do not hold (so the UI never implies more):
  * Set A: the real question. Retrieved node ids, whether the gold chunk was
    among them, how it got there. No answer text was saved for Set A.
  * Set B: the same question with an invented detail inserted. Full answer and
    the model's own sufficiency judgment.
  * Set C: the same question forced through passages that cannot answer it.
    Full answer and judgment.
  * The prune / rehydrate narration comes from the run logs, not from stored
    events, so it is coarser than what a live run streams.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "evaluation" / "results" / "asepsis100"
INDEX_DIR = ROOT / "index"

_EXCERPT_CHARS = 900
_ANSWER_KEYS = ["SHORT_ANSWER", "RECOMMENDED_ACTION", "RATIONALE",
                "LIMITATIONS", "EVIDENCE_SUFFICIENT", "STILL_NEEDED"]

_cache: dict = {"sig": None, "data": None}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _signature() -> tuple:
    paths = [RESULTS_DIR / f"set{s}.json" for s in "ABC"] + sorted(INDEX_DIR.glob("*.json"))
    return tuple((p.name, p.stat().st_mtime_ns) for p in paths if p.exists())


def _walk(nodes, doc, out) -> None:
    for n in nodes or []:
        pin = n.get("pin") or {}
        out[(doc, n["nodeId"])] = {
            "title": n.get("title") or n["nodeId"],
            "summary": n.get("summary") or "",
            "content": n.get("content") or "",
            "page": pin.get("page"),
            "is_leaf": bool(n.get("isLeaf")),
        }
        _walk(n.get("children"), doc, out)


def _load_nodes() -> tuple[dict, dict]:
    """(doc, node_id) -> node, and node_id -> [docs that contain it]."""
    nodes: dict = {}
    for path in sorted(INDEX_DIR.glob("*.json")):
        raw = _read_json(path)
        _walk(raw["nodes"] if isinstance(raw, dict) else raw, path.stem, nodes)
    by_id: dict = {}
    for (doc, nid) in nodes:
        by_id.setdefault(nid, []).append(doc)
    return nodes, by_id


def _parse_answer(text: str) -> dict:
    """SHORT_ANSWER: ... / STILL_NEEDED: ... blocks -> {short_answer: ..., ...}."""
    text = text or ""
    marks = [(m.start(), m.end(), m.group(1))
             for m in re.finditer(r"^(%s):[ \t]*" % "|".join(_ANSWER_KEYS), text, re.M)]
    out = {}
    for i, (_, end, key) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out[key.lower()] = text[end:stop].strip()
    return out


_RESULT_LINE = re.compile(r"^\[\s*\d+/\d+\s+\d+s\]\s+([ABC]\d+)\b")
_PRUNE = re.compile(r"^\s*\[prune\]\s+(\S+): (\d+)/(\d+) leaves after pruning")
_REHYDRATE = re.compile(r"^\s*\[rehydrate\]\s+(\S+): (\d+) leaf\(s\) re-queued")
_PRUNE_FAIL = re.compile(r"^\s*\[prune\] selection failed")


def _parse_steps(log: Path) -> dict:
    """Per-query narration lines from a run log. Lines precede their result line."""
    if not log.exists():
        return {}
    steps: dict = {}
    buf: list = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _RESULT_LINE.match(line)
        if m:
            steps[m.group(1)] = buf
            buf = []
            continue
        if _PRUNE_FAIL.match(line):
            buf.append("One pruning call returned unparseable output — that branch was kept whole")
            continue
        m = _PRUNE.match(line)
        if m:
            doc, kept, total = m.group(1), int(m.group(2)), int(m.group(3))
            buf.append(f"Pruned “{_short(_doc_label(doc))}”: {kept} of {total} sections survive")
            continue
        m = _REHYDRATE.match(line)
        if m:
            doc, n = m.group(1), int(m.group(2))
            buf.append(f"Lexical rehydration re-queued {n} section{'s' if n != 1 else ''} "
                       f"in “{_short(_doc_label(doc))}”")
    return steps


def _doc_label(stem: Optional[str]) -> str:
    """A readable name from the file stem. The manifest's titles are whatever
    the first heading was (the GLASS report's is "Foreword"), so they make poor
    labels."""
    if not stem:
        return ""
    s = re.sub(r"^[0-9a-f]{8}_", "", stem)
    s = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", s).replace("_", " ")
    s = re.sub(r"^WHO Guidelines on ", "", s)
    s = s.replace("Global Antimicrobial Resistance and Use Surveillance System GLASS", "GLASS")
    s = s.replace("Infection Prevention and Control", "IPC")
    return s


def _clean_excerpt(text: str, title: str) -> str:
    """Drop a leading repeat of the heading and collapse blank runs."""
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    base = re.sub(r"\s+—\s+Overview$", "", title).strip().lower()
    while lines and (not lines[0] or lines[0].lstrip("# ").strip().lower() == base):
        lines.pop(0)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _short(title: str, n: int = 64) -> str:
    return title if len(title) <= n else title[: n - 1].rstrip() + "…"


def _build() -> dict:
    nodes, by_id = _load_nodes()

    def passage(node_id: str, prefer: Optional[str] = None) -> dict:
        docs = by_id.get(node_id, [])
        doc = prefer if (prefer, node_id) in nodes else (docs[0] if docs else prefer)
        n = nodes.get((doc, node_id)) or {}
        content = _clean_excerpt(n.get("content") or "", n.get("title") or "")
        return {
            "doc": doc,
            "doc_title": _doc_label(doc),
            "node_id": node_id,
            "title": n.get("title") or node_id,
            "summary": n.get("summary") or "",
            "excerpt": content[:_EXCERPT_CHARS] + ("…" if len(content) > _EXCERPT_CHARS else ""),
            "page": n.get("page"),
        }

    A = {r["id"]: r for r in _read_json(RESULTS_DIR / "setA.json")}
    B = {r["pair_id"]: r for r in _read_json(RESULTS_DIR / "setB.json")}
    C = {r["pair_id"]: r for r in _read_json(RESULTS_DIR / "setC.json")}
    steps_a = _parse_steps(RESULTS_DIR / "setA.log")

    cases: dict = {}
    for aid, a in A.items():
        gold = a["gold_node"]
        retrieved = []
        for nid in a.get("retrieved", []):
            p = passage(nid, a["doc"])
            p["is_gold"] = nid in (a.get("matched") or []) or nid == gold
            retrieved.append(p)

        b = B.get(aid)
        c = C.get(aid)
        case = {
            "id": aid,
            "question": a["question"],
            "doc": a["doc"],
            "doc_title": _doc_label(a["doc"]),
            "gold": passage(gold, a["doc"]),
            "a": {
                "hit": bool(a.get("hit")),
                "how": a.get("how"),
                "reason": (a.get("target_reason") or [None])[0],
                "secs": a.get("secs"),
                "steps": steps_a.get(aid, []),
                "retrieved": retrieved,
            },
            "b": None,
            "c": None,
        }
        if b:
            case["b"] = {
                "id": b["id"],
                "question": b["question"],
                "edit_term": b.get("edit_term"),
                "gold_retrieved": bool(b.get("gold_retrieved")),
                "judgment": b.get("judgment_label"),
                "still_needed": b.get("still_needed") or [],
                "grounding": b.get("grounding_status"),
                "cited": b.get("cited") or [],
                "names_edit_term": bool(b.get("names_edit_term_in_still_needed")),
                "answer": _parse_answer(b.get("answer", "")),
                "passages": [passage(nid, a["doc"]) for nid in b.get("retrieved", [])],
                "secs": b.get("secs"),
            }
        if c:
            case["c"] = {
                "id": c["id"],
                "judgment": c.get("judgment_label"),
                "still_needed": c.get("still_needed") or [],
                "grounding": c.get("grounding_status"),
                "cited": c.get("cited") or [],
                "passed": bool(c.get("passed")),
                "answer": _parse_answer(c.get("answer", "")),
                "passages": [passage(d["node_id"], d.get("doc")) for d in c.get("decoys", [])],
                "secs": c.get("secs"),
            }
        cases[aid] = case
    return cases


def _cases() -> dict:
    sig = _signature()
    if _cache["sig"] != sig:
        _cache["data"] = _build()
        _cache["sig"] = sig
    return _cache["data"]


def available() -> bool:
    return all((RESULTS_DIR / f"set{s}.json").exists() for s in "ABC")


def list_cases() -> list:
    """One row per question, cheap enough to send whole."""
    rows = []
    for aid in sorted(_cases()):
        c = _cases()[aid]
        rows.append({
            "id": aid,
            "question": c["question"],
            "doc_title": c["doc_title"],
            "hit": c["a"]["hit"],
            "n_retrieved": len(c["a"]["retrieved"]),
            "judgment": c["b"]["judgment"] if c["b"] else None,
            "forced_passed": c["c"]["passed"] if c["c"] else None,
            "secs": round(sum(x for x in (
                c["a"]["secs"], c["b"] and c["b"]["secs"], c["c"] and c["c"]["secs"]) if x)),
        })
    return rows


def get_case(case_id: str) -> Optional[dict]:
    return _cases().get(case_id)
