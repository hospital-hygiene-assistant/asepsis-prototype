"""
The evaluation tooling, offline: record derivation from a chat payload, the
four cells, pair drift, the BM25 baseline over a tiny index, the sampler and
validator, the run log, and the report over synthetic records.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import pageindex  # noqa: E402
import runlog  # noqa: E402
from evaluation import bm25_baseline, common, forced_pairing, report, sample_gold  # noqa: E402
from evaluation import validate_queries  # noqa: E402

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# A payload shaped like /api/chat's response
# ---------------------------------------------------------------------------

def _payload(*, retrieved, gold_meta, answer, judgment=None, timing=None, budget=None):
    """retrieved: list of (doc, node_id); gold_meta: node_meta for the gold."""
    sources = [{"n": i + 1, "id": f"s{i + 1}", "doc": d, "node_id": n,
                "title": n, "breadcrumb": "", "excerpt": "..."}
               for i, (d, n) in enumerate(retrieved)]
    node_meta = {n: {"status": "retrieved", "bm25_rank": i} for i, (_, n) in enumerate(retrieved)}
    node_meta.update(gold_meta)
    results = {"guideline": {"tree": [], "retrieved_ids": [n for _, n in retrieved],
                             "node_meta": node_meta, "nodes": []}}
    import synthesis
    sections = synthesis.parse_answer_sections(answer)
    return {
        "query": "q", "run_id": "r1",
        "answer": {"content": answer, **sections},
        "grounding": {"status": synthesis.grounding_status(answer, len(sources))[0],
                      "summary": "", "sources": sources},
        "judgment": judgment if judgment is not None else synthesis.interpret_judgment(sections),
        "timing": timing or {"retrieval_ms": 1000, "prune_ms": 400, "evaluate_ms": 600,
                             "synthesis_ms": 2000, "total_ms": 3000},
        "budget": budget or {"evaluated": 5, "deferred": 0, "capped_by": "",
                             "tokens_used": 10, "tokens_max": 100, "deferred_nodes": []},
        "run": {"results": results},
    }


def _q(id="A001", set="A", pair_id="", node="gold", edit_term=""):
    return common.Query(id=id, set=set, pair_id=pair_id, doc="guideline",
                        node_id=node, question="q?", edit_term=edit_term)


class TestDeriveRecord:
    def test_gold_retrieved_and_cited_and_sufficient(self):
        p = _payload(retrieved=[("guideline", "other"), ("guideline", "gold")],
                     gold_meta={"gold": {"status": "retrieved", "bm25_rank": 3}},
                     answer="SHORT_ANSWER: x [2].\nEVIDENCE_SUFFICIENT: yes")
        r = common.derive_record(_q(), p, index_fingerprint="fp")
        assert r.gold_retrieved and r.gold_cited
        assert r.gold_outcome == "retrieved" and r.gold_bm25_rank == 3
        assert r.judgment_label == "sufficient"
        assert r.cell == "retrieved_sufficient"
        assert r.grounding_status == "grounded"
        assert r.timing["synthesis_ms"] == 2000
        assert r.index_fingerprint == "fp"

    def test_gold_retrieved_but_not_cited_is_the_recall_gap(self):
        p = _payload(retrieved=[("guideline", "other"), ("guideline", "gold")],
                     gold_meta={}, answer="SHORT_ANSWER: x [1].\nEVIDENCE_SUFFICIENT: yes")
        r = common.derive_record(_q(), p)
        assert r.gold_retrieved and not r.gold_cited

    def test_missed_and_sufficient_is_the_confidently_wrong_cell(self):
        p = _payload(retrieved=[("guideline", "other")],
                     gold_meta={"gold": {"status": "rejected", "bm25_rank": 1}},
                     answer="SHORT_ANSWER: x [1].\nEVIDENCE_SUFFICIENT: yes")
        r = common.derive_record(_q(), p)
        assert r.cell == "missed_sufficient"
        assert r.gold_outcome == "rejected" and r.gold_bm25_rank == 1

    def test_pruned_gold_has_no_rank(self):
        p = _payload(retrieved=[], gold_meta={"gold": {"status": "pruned"}},
                     answer="")
        r = common.derive_record(_q(), p)
        assert r.gold_outcome == "pruned" and r.gold_bm25_rank is None
        assert r.grounding_status == "insufficient_evidence"
        assert r.cell == "missed_unlabeled"

    def test_deferred_gold_is_censored_not_missed(self):
        p = _payload(retrieved=[("guideline", "other")],
                     gold_meta={"gold": {"status": "deferred", "bm25_rank": 40}},
                     answer="SHORT_ANSWER: x [1].\nEVIDENCE_SUFFICIENT: yes")
        r = common.derive_record(_q(), p)
        assert r.gold_outcome == "deferred"
        assert r.cell == "deferred_sufficient"

    def test_deferred_but_judged_relevant_is_its_own_outcome(self):
        p = _payload(retrieved=[("guideline", "other")],
                     gold_meta={"gold": {"status": "deferred", "bm25_rank": 2,
                                         "judged_relevant": True}},
                     answer="SHORT_ANSWER: x [1].\nEVIDENCE_SUFFICIENT: yes")
        r = common.derive_record(_q(), p)
        assert r.gold_outcome == "deferred_judged_relevant"
        assert r.cell == "missed_sufficient", "a positive verdict cut by budget is a miss, not censored"

    def test_missing_label_is_unlabeled_not_sufficient(self):
        p = _payload(retrieved=[("guideline", "gold")], gold_meta={},
                     answer="SHORT_ANSWER: x [1].")
        r = common.derive_record(_q(), p)
        assert r.judgment_label == "missing" and r.judgment_raw is None
        assert r.cell == "retrieved_unlabeled"
        assert r.judgment_ui_reading == "sufficient"

    def test_set_b_names_edit(self):
        answer = ("SHORT_ANSWER: x [1].\nEVIDENCE_SUFFICIENT: no\n"
                  "STILL_NEEDED: - guidance on Mucus Memory Syndrome")
        p = _payload(retrieved=[("guideline", "gold")], gold_meta={}, answer=answer)
        r = common.derive_record(_q(id="B001", set="B", pair_id="A001",
                                    edit_term="Mucus Memory Syndrome"), p)
        assert r.cell == "retrieved_insufficient"
        assert r.names_edit is True
        r2 = common.derive_record(_q(id="B001", set="B", pair_id="A001",
                                     edit_term="Phantom Rhinitis"), p)
        assert r2.names_edit is False

    def test_error_payload(self):
        r = common.derive_record(_q(), {"error": "HTTP 500: boom"})
        assert r.error and r.cell == "error"

    def test_judgment_interpreted_locally_when_server_omits_it(self):
        p = _payload(retrieved=[("guideline", "gold")], gold_meta={},
                     answer="SHORT_ANSWER: x [1].\nEVIDENCE_SUFFICIENT: no")
        del p["judgment"]
        assert common.derive_record(_q(), p).judgment_label == "insufficient"


class TestPairDrift:
    def test_same_and_shifted(self):
        a = {"retrieved": [["d", "1"], ["d", "2"]], "gold_retrieved": True}
        b = {"retrieved": [["d", "2"], ["d", "1"]], "gold_retrieved": True}
        d = common.pair_drift(a, b)
        assert d["same_set"] and d["jaccard"] == 1.0 and d["gold_in_both"]
        c = {"retrieved": [["d", "1"], ["d", "3"]], "gold_retrieved": False}
        d = common.pair_drift(a, c)
        assert not d["same_set"] and d["only_b"] == [("d", "3")] and not d["gold_in_both"]


class TestPercentile:
    def test_nearest_rank(self):
        assert common.percentile([], 50) is None
        assert common.percentile([5], 95) == 5
        vals = list(range(1, 101))
        assert common.percentile(vals, 50) == 50
        assert common.percentile(vals, 95) == 95


# ---------------------------------------------------------------------------
# Corpus-backed tools, over the conftest sample document
# ---------------------------------------------------------------------------

@pytest.fixture
def built_corpus(tmp_corpus, fake_ollama):
    fake_ollama.default_response = "a summary"
    pageindex.build_index("guideline", use_llm_summaries=False)
    return tmp_corpus


class TestCorpusTools:
    def test_leaves_in_server_order_with_breadcrumbs(self, built_corpus):
        leaves = common.load_corpus_leaves(built_corpus["index"])
        ids = [l.node_id for l in leaves]
        assert "first-line-therapy" in ids and "imaging" in ids
        assert [l.order for l in leaves] == list(range(len(leaves)))
        first_line = next(l for l in leaves if l.node_id == "first-line-therapy")
        assert "Treatment" in first_line.breadcrumb
        assert first_line.chunk_id == "guideline/first-line-therapy"

    def test_bm25_baseline_finds_the_gold(self, built_corpus, tmp_path):
        qfile = tmp_path / "q.csv"
        common.write_queries(qfile, [
            common.Query("A001", "A", "", "guideline", "first-line-therapy",
                         "amoxicillin dose for first-line therapy"),
            common.Query("A002", "A", "", "guideline", "imaging",
                         "is CT preferred over ultrasound"),
        ])
        out = tmp_path / "bm25"
        rc = bm25_baseline.main(["--queries", str(qfile), "--out", str(out),
                                 "--index-dir", str(built_corpus["index"]), "--ks", "1,3"])
        assert rc == 0
        rows = common.read_jsonl(out / "bm25.jsonl")
        assert all(r["gold_rank"] == 1 for r in rows), rows
        summary = json.loads((out / "bm25_summary.json").read_text())
        assert summary["all"]["recall@1"] == 1.0

    def test_budget_matched_k_from_a_run(self, tmp_path):
        run = tmp_path / "run"
        for n in (3, 5, 4):
            common.append_jsonl(run / "records.jsonl", {"id": f"A{n}", "n_retrieved": n})
        b = bm25_baseline.budget_matched_k(run)
        assert b["k"] == 4 and b["n_runs"] == 3

    def test_sampler_writes_pairs_and_sidecar(self, built_corpus, tmp_path):
        out = tmp_path / "q" / "queries.csv"
        rc = sample_gold.main(["--n", "2", "--seed", "3", "--out", str(out),
                               "--index-dir", str(built_corpus["index"]), "--min-chars", "10"])
        assert rc == 0
        qs = common.load_queries(out)
        assert [q.set for q in qs] == ["A", "B", "A", "B"]
        assert qs[1].pair_id == qs[0].id and qs[1].gold == qs[0].gold
        assert out.with_suffix(".chunks.md").exists()

    def test_validator_catches_bad_gold_and_bad_pairs(self, built_corpus, tmp_path):
        qfile = tmp_path / "q.csv"
        common.write_queries(qfile, [
            common.Query("A001", "A", "", "guideline", "first-line-therapy", "q1"),
            common.Query("B001", "B", "A001", "guideline", "imaging", "q1 edited", ""),
            common.Query("A002", "A", "", "guideline", "no-such-leaf", "q2"),
            common.Query("B003", "B", "A999", "guideline", "imaging", "q3"),
        ])
        errors, warnings = validate_queries.validate(qfile, built_corpus["index"])
        joined = "\n".join(errors)
        assert "B001: gold differs" in joined
        assert "A002: gold guideline/no-such-leaf" in joined
        assert "B003: pair_id 'A999'" in joined
        assert any("no edit_term" in w for w in warnings)

    def test_forced_pairing_gold_assignment_never_hands_back_own_gold(self, built_corpus):
        qs = [common.Query("A001", "A", "", "guideline", "first-line-therapy", "q1"),
              common.Query("A002", "A", "", "guideline", "imaging", "q2"),
              common.Query("A003", "A", "", "guideline", "initial-workup", "q3")]

        class Args:
            source, k, from_run, document = "gold", 2, None, None

        assignments = forced_pairing.build_assignments(qs, Args)
        assert len(assignments) == 3
        for q, sources, _ in assignments:
            assert sources and all(s["node_id"] != q.node_id for s in sources)
            assert [s["n"] for s in sources] == list(range(1, len(sources) + 1))

    def test_forced_pairing_document_chunks(self, tmp_path):
        doc = tmp_path / "ctrl.md"
        doc.write_text("# T\n\n## One\nalpha\n\n## Two\nbeta\n", encoding="utf-8")
        chunks = forced_pairing._document_chunks(doc)
        assert [c["title"] for c in chunks] == ["One", "Two"]


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------

class TestRunLog:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv(runlog.ENV_VAR, raising=False)
        assert not runlog.enabled()
        assert runlog.append({"x": 1}) is False

    def test_record_shape_and_append(self, monkeypatch, tmp_path):
        monkeypatch.setenv(runlog.ENV_VAR, str(tmp_path / "runs.jsonl"))
        results = {"doc": {"node_meta": {"a": {"status": "retrieved", "bm25_rank": 0, "ms": 5},
                                         "b": {"status": "deferred", "bm25_rank": 9,
                                               "judged_relevant": True}}}}
        sources = [{"n": 1, "doc": "doc", "node_id": "a"}]
        rec = runlog.build_record(kind="chat", query="q", results=results,
                                  budget={"deferred_nodes": [{"doc": "doc", "node_id": "b",
                                                              "bm25_rank": 9}],
                                          "evaluated": 2},
                                  timing={"retrieval_ms": 10}, index_dir=tmp_path,
                                  sources=sources, grounding_status="grounded",
                                  judgment={"raw": "yes", "label": "sufficient"}, cited=[1])
        assert rec["node_statuses"]["doc"]["b"]["judged_relevant"] is True
        assert rec["cited"] == [{"doc": "doc", "node_id": "a", "n": 1}]
        assert "deferred_nodes" not in rec["budget"] and rec["deferred_nodes"][0]["node_id"] == "b"
        assert rec["config"]["completeness_check"] in (True, False)
        assert runlog.append(rec) is True
        lines = (tmp_path / "runs.jsonl").read_text().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["judgment"]["raw"] == "yes"


class TestRunContextTimings:
    def test_timed_phases_survive_snapshot_and_restore(self):
        ctx = pageindex.new_run(make_current=False)
        with ctx.timed("prune"):
            pass
        assert "prune" in ctx.timings and ctx.timings["prune"] >= 0
        snap = ctx.events_snapshot()
        assert snap["timings"] == ctx.timings
        other = pageindex.new_run(make_current=False)
        other.restore(snap)
        assert other.timings == ctx.timings


# ---------------------------------------------------------------------------
# Report over synthetic records
# ---------------------------------------------------------------------------

def _record(id, set, cell, outcome, retrieved=True, cited=True, label=None, pair_id="",
            rank=None, retrieved_ids=(("guideline", "gold"),), names_edit=None):
    label = label or cell.split("_", 1)[1]
    return {
        "id": id, "set": set, "pair_id": pair_id, "query": "q", "gold_doc": "guideline",
        "gold_node_id": "gold", "edit_term": "", "run_id": "r",
        "n_retrieved": len(retrieved_ids), "retrieved": [list(x) for x in retrieved_ids],
        "gold_retrieved": retrieved, "gold_outcome": outcome, "gold_bm25_rank": rank,
        "gold_bm25_score": None, "grounding_status": "grounded" if cited else "partially_grounded",
        "cited": [["guideline", "gold"]] if cited else [], "gold_cited": cited,
        "judgment_raw": {"sufficient": "yes", "insufficient": "no"}.get(label),
        "judgment_label": label, "judgment_ui_reading": "sufficient" if label != "insufficient" else "insufficient",
        "still_needed": ["x"] if label == "insufficient" else [], "names_edit": names_edit,
        "cell": cell, "timing": {"retrieval_ms": 100, "prune_ms": 40, "evaluate_ms": 60,
                                 "synthesis_ms": 50, "total_ms": 150},
        "budget": {"capped_by": "context" if outcome.startswith("deferred") else ""},
        "index_fingerprint": "fp", "error": "", "answer_text": "", "extra": {},
    }


class TestReport:
    def test_report_computes_every_section(self, tmp_path):
        run = tmp_path / "main"
        recs = [
            _record("A001", "A", "retrieved_sufficient", "retrieved", rank=0),
            _record("A002", "A", "missed_sufficient", "rejected", retrieved=False, cited=False, rank=1),
            _record("A003", "A", "deferred_sufficient", "deferred", retrieved=False, cited=False, rank=30),
            _record("A004", "A", "retrieved_unlabeled", "retrieved", label="missing"),
            _record("B001", "B", "retrieved_insufficient", "retrieved", pair_id="A001", names_edit=True),
            _record("B002", "B", "missed_insufficient", "pruned", retrieved=False, cited=False,
                    pair_id="A002", retrieved_ids=(("guideline", "other"),)),
        ]
        for r in recs:
            common.append_jsonl(run / "records.jsonl", r)
        (run / "manifest.json").write_text(json.dumps({"arm": "main", "config": {"agent_ctx": 1}}))

        raised = tmp_path / "raised"
        common.append_jsonl(raised / "records.jsonl",
                            _record("A003", "A", "retrieved_sufficient", "retrieved", rank=30))

        text, data = report.build(run, raised, None, None, None)
        assert data["cells_A"] == {"retrieved_sufficient": 1, "missed_sufficient": 1,
                                   "deferred_sufficient": 1, "retrieved_unlabeled": 1}
        assert data["cells_B"] == {"retrieved_insufficient": 1, "missed_insufficient": 1}
        # deferred excluded from recall denominators; unlabeled still counts for retrieval
        assert data["recall"]["retrieval_recall"] == [3, 5]
        assert data["recall"]["answer_recall"] == [3, 5]
        assert data["death"]["outcomes"]["deferred"] == 1
        assert data["pairs"]["same"] == 1 and data["pairs"]["shifted"] == 1
        assert data["pairs"]["transitions"]["sufficient→insufficient"] == 2
        assert data["budget"]["recovered"] == 1 and data["budget"]["rerun"] == 1
        assert data["latency"]["all.synthesis_ms"]["p50"] == 50
        assert data["judgment"]["silently_sufficient"] == ["A004"]
        assert "confidently wrong" in text
        assert "diagnostic only" in text

    def test_report_cli_writes_files(self, tmp_path):
        run = tmp_path / "main"
        common.append_jsonl(run / "records.jsonl",
                            _record("A001", "A", "retrieved_sufficient", "retrieved"))
        assert report.main(["--run", str(run)]) == 0
        assert (run / "report.md").exists() and (run / "report.json").exists()


class TestDriverSelection:
    def test_only_gold_deferred_selects_both_deferred_kinds(self, tmp_path):
        from evaluation import run_eval
        prior = tmp_path / "main"
        for id, outcome in [("A1", "retrieved"), ("A2", "deferred"),
                            ("A3", "deferred_judged_relevant"), ("A4", "pruned")]:
            common.append_jsonl(prior / "records.jsonl", {"id": id, "gold_outcome": outcome})
        qs = [common.Query(i, "A", "", "d", "n", "q") for i in ("A1", "A2", "A3", "A4")]
        assert [q.id for q in run_eval.select_queries(qs, prior)] == ["A2", "A3"]

    def test_records_roundtrip_through_asdict(self):
        p = _payload(retrieved=[("guideline", "gold")], gold_meta={},
                     answer="SHORT_ANSWER: x [1].\nEVIDENCE_SUFFICIENT: yes")
        r = common.derive_record(_q(), p)
        d = asdict(r)
        assert d["cell"] == "retrieved_sufficient" and json.dumps(d)
