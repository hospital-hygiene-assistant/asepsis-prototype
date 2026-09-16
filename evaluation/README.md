# Evaluating the retrieval system

This directory holds everything needed to run the evaluation once the corpus
(six hand-written markdown documents) and the question sets are in place.
Nothing here needs the desktop shell; the backend is a plain HTTP server.

## What is measured, in one paragraph

Two independent signals come out of every answer and are never conflated.
The **grounding status** is mechanical — no evidence at all, cited, or
uncited — and needs no evaluation. The **model's own judgment** about whether
the retrieved passages were sufficient (`EVIDENCE_SUFFICIENT`) and what it
says is still needed (`STILL_NEEDED`) is what gets evaluated, scored
*conditional on retrieval succeeding*. Alongside: retrieval recall vs answer
recall, where the gold chunk died when it was missed, whether the paired edits
moved retrieval, the cost of the context budget in recall points, and latency
per phase. A BM25-only lower bound and a frontier-model upper bound frame the
result.

## Files

| File | What it does |
|---|---|
| `sample_gold.py` | Random sample of N leaves → query CSV skeleton + a sidecar with each chunk's text |
| `validate_queries.py` | Checks gold chunks exist, pairs align, ids are unique |
| `run_eval.py` | Drives `POST /api/chat` over Sets A and B; saves every payload; resumable |
| `forced_pairing.py` | Set C: synthesis only, over passages that cannot answer the question |
| `bm25_baseline.py` | BM25 over every leaf, offline, no LLM |
| `frontier_baseline.py` | Whole corpus in a cached prefix; asks for chunk ids, not answers |
| `report.py` | Aggregates a run directory into `report.md` / `report.json` |
| `common.py` | Query format, corpus loading, and the per-query record derivation |
| `queries/example_queries.csv` | Four A/B pairs against the sample corpus, for smoke tests |
| `queries/nonmedical_control.md` | A non-medical document for the forced-pairing control |

Supporting changes outside this directory:

- `synthesis.py` — the synthesis prompt, passage format, answer parser and the
  call itself, shared by the server and these scripts. The chat handler no
  longer builds the prompt inline.
- `runlog.py` — optional JSONL log of every run the server makes, enabled by
  `ASEPSIS_RUN_LOG=/path/runs.jsonl`.
- `pageindex.RunContext.timings` and the `timing` field on `/api/chat` and
  `/api/run`: wall-clock per phase (`prune`, `evaluate`, `retrieval`,
  `synthesis`).
- `judgment` field on `/api/chat`: the raw `EVIDENCE_SUFFICIENT` string, a
  strict interpretation, and what the product's own prefix match would read.
- `fingerprint` on `/api/index/state`: a content hash of the index, so a
  mid-run re-index is detectable.

## The query file

```
id,set,pair_id,doc,node_id,question,edit_term
A017,A,,who_hand_hygiene,handrub-duration,How long should an alcohol handrub take?,
B017,B,A017,who_hand_hygiene,handrub-duration,How long should an alcohol handrub take for a patient with Phantom Rhinitis?,Phantom Rhinitis
```

- `doc` is the index stem (`index/<doc>.json`); `node_id` is the leaf's
  `nodeId`. The gold label is a chunk, not a page.
- A Set B row shares its A partner's gold chunk. `edit_term` is the invented
  detail that was inserted, so the report can check whether `STILL_NEEDED`
  names it.
- The earlier `questions.csv` (`document,page,question`) was page-based and
  cannot be mapped onto hand-written markdown; write the new sets against the
  sampled chunks instead.

## Running it, in order

```bash
# 0. corpus in docs/, ingested and indexed as usual (pipeline.py ingest / index)

# 1. sample gold chunks, then write the questions into the CSV
python3 -m evaluation.sample_gold --n 100 --seed 1 --out evaluation/queries/queries.csv
python3 -m evaluation.validate_queries evaluation/queries/queries.csv

# 2. start the backend headless (no desktop shell), with the run log on
ASEPSIS_RUN_LOG=evaluation/results/runs.jsonl python3 tauri-app/server.py

# 3. main arm at production settings
python3 -m evaluation.run_eval --queries evaluation/queries/queries.csv \
    --out evaluation/results/main

# 4. raised-cap arm — only the queries whose gold chunk was deferred
python3 -m evaluation.run_eval --queries evaluation/queries/queries.csv \
    --out evaluation/results/raised --arm raised \
    --only-gold-deferred-from evaluation/results/main \
    --agent-ctx 131072 --max-leaf-evals 2000

# 5. Set C, against the passages the system retrieved for OTHER questions
python3 -m evaluation.forced_pairing --queries evaluation/queries/queries.csv \
    --out evaluation/results/forced --source derange --from-run evaluation/results/main

# 6. baselines
python3 -m evaluation.bm25_baseline --queries evaluation/queries/queries.csv \
    --out evaluation/results/bm25 --budget-from evaluation/results/main
python3 -m evaluation.frontier_baseline --queries evaluation/queries/queries.csv \
    --out evaluation/results/frontier          # dry run: prints tokens and cost
python3 -m evaluation.frontier_baseline ... --run   # when ready to spend

# 7. the report
python3 -m evaluation.report --run evaluation/results/main \
    --raised evaluation/results/raised --bm25 evaluation/results/bm25 \
    --frontier evaluation/results/frontier --forced evaluation/results/forced
```

### Conditions the driver enforces

- The multiple-choice pre-filter is off (no `selected_answers` are sent). It
  materially changes retrieval and belongs in its own arm if evaluated at all.
- The retrieval cache is off (`use_cache: false` on every request). The debug
  cache may be *enabled* on the server — it only records — but a measured run
  never replays.
- `completeness_check` must be on, or the judgment is never emitted; the driver
  refuses to start otherwise.
- The index must not be stale, and no other run may be in flight.
- The index fingerprint is re-read after every query. If it changes, the run
  stops. Do not re-index mid-run.
- The gold chunk is never injected into the context. The raised-cap arm raises
  `agent_ctx` and `max_leaf_evals` through `/api/config` for the duration of
  the run and restores them after.

### The debug cache, for one purpose

Enable it (`POST /api/config {"debug_cache_enabled": true}`) when iterating on
the synthesis prompt: it freezes a retrieval result so the synthesis step can
be re-run against an unchanged passage set (`use_cache: true` on the request
replays retrieval and re-synthesises live). Its key includes the index
fingerprint and the prompt versions, so a re-index or a prompt bump
invalidates it correctly. Keep it out of measured runs.

## How the numbers are computed

Every `/api/chat` payload is saved whole under `raw/<id>.json`, and
`common.derive_record` reduces it to one flat record. The report is computed
from those records only.

**Gold outcome.** The gold leaf's terminal status from the run's `node_meta`:
`retrieved`, `rejected` (evaluator read it and said no), `pruned` (a section
above it was cut before the leaf was ever read), `deferred` (a budget was hit
first), `deferred_judged_relevant`, `error`. The last-but-one exists because
the system marks the boundary node — evaluated, judged relevant, then found
not to fit — as `deferred` too; it is a positive verdict cut by the budget,
not a censored one, so it is scored as a miss rather than excluded.

**The four cells**, per set, conditional on retrieval and excluding deferred
gold and unlabeled answers:

| | model: sufficient | model: insufficient |
|---|---|---|
| gold retrieved | correct (A) / false alarm (B) | false alarm (A) / correct (B) |
| gold missed | **confidently wrong** | failing safe |

Set A should land in *retrieved + sufficient*; Set B in *retrieved +
insufficient*, naming the missing detail. The *missed + sufficient* cell is
reported prominently and separately.

**The judgment label** is kept raw. The product reads it with a loose prefix
match (`/^\s*no\b/i`) and treats an absent label as sufficient; the record
carries the raw string, a strict reading (`sufficient` / `insufficient` /
`unparsed` / `missing`), and the product's reading, so both failure modes are
visible rather than swallowed.

**Retrieval recall vs answer recall.** Gold in the retrieved set, and gold
cited inline in the finished answer. The second lags the first with a large
retrieved set, and the gap is a finding.

**BM25 rank.** The rank recorded inside a run is among post-pruning survivors,
corpus-wide. The baseline's rank is among every leaf in the corpus. Both are
reported; pruned gold has no in-run rank.

**Pairs.** For each A/B pair the two retrieved sets are compared. Pairs where
the set was unchanged are the clean comparison; shifted pairs are reported
separately.

**Budget.** `deferred` is censored and never folded into `missed`. The
raised-cap arm re-runs only the deferred-gold queries; the number recovered
into the retrieved set is the cost of the production budget in recall points.
That arm is diagnostic only: `config.py` puts the model's quality band at
32–64k with a 48k default, well below its 128k maximum.

**Latency.** p50/p95 per phase, retrieval and synthesis separately, from the
`timing` field. Retrieval is further split into `prune` and `evaluate`.

## Baselines

**BM25 alone** (lower bound). The system ranks its post-pruning candidates with
this same BM25 and then spends one model call per leaf, accepting or rejecting
in that order; the question is whether the LLM layer improves on the ranking
it is built on. Budget-matched: k equals the mean number of passages the
system actually retrieved in the main arm.

**Frontier model, whole corpus in context** (upper bound). Same task: every
chunk is labelled with its id and the model names the ids it would cite,
rather than answering. The corpus is a cached prefix (1-hour TTL); only the
question varies after the breakpoint. The prefix contains no timestamps or
per-query ids and its SHA-256 is recorded. Cache hits are verified from
`usage.cache_read_input_tokens` on every response, and the run aborts after
repeated misses. The default invocation is a dry run that counts tokens and
prints an estimated cost; nothing is sent until `--run`. A ceiling cannot be
used clinically, so the system does not need to beat it.

**Forced pairing** (Set C). Synthesis only, over passages that cannot answer
the question. The model should say insufficient every time. A cheap floor.

## Out of scope, and one bias to record

- Ingestion fidelity: hand-written markdown, no parser to evaluate.
- Test–retest variance: each query runs once; accepted as a limitation.
- Stratified sampling; multi-chunk questions.
- **Question-generation leakage.** Questions written from the chunk text share
  its vocabulary. That inflates BM25 and therefore the system built on it,
  while leaving the full-text frontier baseline unaffected. The bias runs in
  the system's favour and matters when reporting how close the system came to
  the ceiling.

## Tests

`python3 run_tests.py --phase 7` runs the offline tests for the synthesis
module and the evaluation tooling (`tests/test_synthesis.py`,
`tests/test_evaluation.py`). They need no Ollama and no server.
