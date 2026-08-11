# Asepsis Prototype — Retrieval Rework Plan

Branch: **`tags-optim-debug`**.

Target model: **`gemma4:e4b`**, 128k max context. Working context defaults to
48k (the 32–64k band where quality holds for complex reasoning) and is
adjustable at runtime from the frontend — see Phase 0.1.

Ordering is dependency-driven: Phase 0 unblocks everything, Phase 1 unblocks
Phase 2, Phase 2+3 must land before Phase 5 (MCQ precompute reuses the prune
machinery and we don't want cached results from the weaker version).

---

## Phase 0 — Foundations

Nothing user-visible; every later phase depends on this.

### 0.1 Model + context configuration

New `config.py` at project root:

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str            # ollama tag
    max_ctx: int         # hardware/model ceiling
    default_ctx: int     # working default — quality band, not the ceiling
    reserve: int         # tokens reserved for the response

MODEL_SPECS = {
    "gemma4:e4b": ModelSpec("gemma4:e4b", max_ctx=131072,
                            default_ctx=49152, reserve=2048),
}
DEFAULT_RETRIEVAL = DEFAULT_SYNTHESIS = "gemma4:e4b"
```

`max_ctx` (128k) is the ceiling; `default_ctx` (48k) is what we actually run,
since the 32–64k band is where quality holds for complex reasoning.

**Runtime-adjustable.** A `RuntimeConfig` holds the live `retrieval_ctx` and
`agent_ctx`, persisted to `config/runtime.json`, surfaced in `GET/POST
/api/config` and clamped to `[4096, spec.max_ctx]`. The frontend gets a
context-window control in the existing settings/inspector surface (a slider
with the 32–64k band marked as recommended), so the agent's working window is
tunable without a restart. Changing `agent_ctx` changes the Phase 3 budget,
so the deferred-node count moves with the slider — which makes the trade-off
directly observable.

- Every `client.chat(...)` call site passes
  `options={"temperature": 0, "num_ctx": spec.num_ctx}`.
  Call sites today: `pageindex._chat:622`, `query._synthesise:57`,
  `server.chat:1154`, `modules/ingest/_captioning.py:101`.
- Unknown model tag → conservative default (`num_ctx=8192`) **plus a loud
  warning**, never a silent 4096.
- `/api/config` and `/api/chat/config` expose the resolved spec so the
  inspector shows the real context window in use.

### 0.2 Token accounting

`tokens.py`:
- `estimate_tokens(text) -> int` — chars/3.6 heuristic, deliberately
  conservative (over-estimates).
- **Calibration:** Ollama responses carry `prompt_eval_count`. After each call
  we record `(estimated, actual)` and expose a rolling ratio in
  `/api/status`, so the heuristic's error is observable rather than assumed.
  If the ratio drifts past a threshold the estimator scales itself.
- `budget_fit(items, max_tokens, overhead)` — greedy accumulator returning
  `(fitted, dropped, tokens_used)`. This is the single primitive Phase 3 uses.

### 0.3 Per-run state (kills the module globals)

Replace `pageindex.py:50-63` globals (`_pruned_ids`, `_retrieved_ids`,
`_kept_ids`, `_rejected_ids`, `_live_meta`, `_prog_*`) with:

```python
@dataclass
class RunContext:
    run_id: str
    lock: threading.Lock
    progress: dict
    events: dict[str, set[str]]     # kept/pruned/retrieved/rejected/deferred/error
    meta: dict[str, dict]
    budget: BudgetState
```

- `RUNS: dict[str, RunContext]` + `CURRENT_RUN` pointer so `/api/status` keeps
  its existing shape (frontend untouched).
- `retrieve_with_metadata(doc, query, *, ctx: RunContext)` — threaded through
  `_prune_and_collect`, `_check_section_relevant`, `_evaluate_leaf`.
- Module-level `start_run/get_progress/get_live_events` become thin wrappers
  over `CURRENT_RUN` for back-compat with `server.py:912,1061`.

This is what makes Phase 5's background precompute safe to run while a query
is in flight.

### 0.4 Bounded, shared executor

`ThreadPoolExecutor(max_workers=len(surviving))` (`pageindex.py:1026`) and
`max_workers=len(sections)` (`:917`) spawn one thread per node.

- One process-wide `WORK_POOL` sized `concurrency_per_instance × len(clients)`
  (default 2 per instance, configurable).
- Both doc-level fan-out (0.5) and node-level fan-out submit to the *same*
  pool, so parallelising documents cannot multiply thread counts.

### 0.5 Parallel documents

`server._run_retrieval:915` loops documents serially. Submit per-doc retrieval
to `WORK_POOL`; merge results as they complete. Progress denominator is
already computed globally up front, so live progress still works.

Also fix: that function parses every `index/*.json` twice (once for the leaf
count at `:908`, once per doc at `:927`). Parse once, reuse.

### 0.6 Error handling

- New node status `error`, distinct from `rejected`. A leaf whose eval raises
  or fails JSON parsing is **not** silently rejected (`pageindex.py:748`).
- One retry with a stricter "output only JSON" reprompt, then `error`.
- Prune-check keeps its conservative keep-on-error behaviour but records
  `error` in meta so the UI can show it.
- `error` gets its own colour in the treemap legend
  (`ui/index.html:240-243, 286-293`).

### 0.7 Dead code

Remove `_flatten_toc` (`pageindex.py:483`) and `_find_nodes_by_ids` (`:517`) —
referenced only by `tests/test_pageindex_units.py`. Drop their test classes.

### Tests — Phase 0
`tests/test_config_context.py`, `tests/test_tokens.py`,
`tests/test_runcontext.py`
- every chat call site receives a `num_ctx` (asserted via a recording fake
  client); unknown model warns and falls back
- estimator monotonic, over-estimates on ASCII+unicode samples;
  `budget_fit` boundary cases (empty, single oversized item, exact fit)
- two concurrent `RunContext`s do not observe each other's events;
  `CURRENT_RUN` wrappers return the active run
- pool never exceeds configured max under a 500-node fan-out
- leaf eval raising → status `error`, not `rejected`; retry path exercised

---

## Phase 1 — Real summaries, at index time, persistent

### 1.1 LLM summarisation, bottom-up

Replace the heuristic `_generate_summary` (`pageindex.py:422`):

- **Leaf** (incl. synthetic overview leaves): `LEAF_SUMMARY_PROMPT` over the
  leaf content → 1–2 sentence factual summary of what the leaf *contains*
  (not whether it answers anything).
- **Internal**: `SECTION_SUMMARY_PROMPT` over the *already-computed child
  summaries* + the section's own overview-leaf text. Bottom-up, so a parent
  never re-reads raw content — cost is O(nodes), one call each.
- Long leaves are truncated to the summariser's budget via `budget_fit`,
  with a `truncated: true` marker on the node.

### 1.2 Overview / synthetic-leaf handling

- Bug fix at `pageindex.py:427-431`: `titles` filters synthetic children but
  `extra = node.children[4:]` slices the unfiltered list. Moot once the
  heuristic is replaced, but the same filtering mistake must not be carried
  into the new prompt-input builder.
- Synthetic overview leaves are *content*, not structure: they are summarised
  as leaves, and their summary is fed into the parent's summary input clearly
  labelled as the section's own prose (currently they're excluded from the
  parent summary entirely, which drops the section's introductory text).

### 1.3 Persistence + incremental rebuild

Per node in `index/<doc>.json`:
`contentHash` (sha256 of the summariser input), `summaryModel`,
`summaryPromptVersion`.

`build_index` loads the existing index when present and **only re-summarises
nodes whose hash/model/prompt-version changed**, propagating upward (a changed
leaf dirties its ancestors). Unchanged docs cost zero LLM calls on re-index.

### 1.4 Graceful degradation

Ollama unreachable at index time → fall back to the old heuristic, mark
`summarySource: "heuristic"`, and surface a warning in the ingest progress
payload. Flagged, never silent.

### Tests — Phase 1
`tests/test_summaries.py`
- fake client: bottom-up order asserted (children summarised before parent);
  parent prompt contains child *summaries*, never raw child content
- synthetic overview leaf is summarised and appears in its parent's input
- rerun with unchanged doc → **zero** LLM calls; edit one leaf → exactly that
  leaf + its ancestor chain re-summarised
- prompt-version bump invalidates everything
- oversized leaf truncated and marked
- client down → heuristic fallback + warning, index still valid

---

## Phase 2 — Batched, summary-driven pruning

### 2.1 One decision call per parent

`_check_section_relevant` (`pageindex.py:802`) and
`_format_descendant_outline` (`:790`) are replaced by `_select_children`:

Input: query, parent breadcrumb, and the parent's **direct children only**,
each as `id | title | SECTION|LEAF | summary`. No recursion into the subtree —
the summaries already carry descendant meaning (that's what Phase 1 bought).

Output: `{"keep": [{"id": ..., "reason": ...}, ...]}`, i.e. a subset chosen
**comparatively**, in one call for the whole sibling set.

- Traversal stays BFS: kept sections join the next frontier, kept leaves join
  the candidate queue, unkept sections are pruned with their whole subtree
  (existing `_prune_and_collect` bookkeeping at `:943-962` is preserved).
- **Leaves are never batch-judged here** — a kept leaf is a *candidate*, and
  its real verdict comes from Phase 3's per-leaf evaluator.
- Wide fan-out: if the children list exceeds the prune budget, chunk into
  batches of K and union the kept sets (recorded as `batched: true` so the
  loss of full comparative context is visible).
- Hallucinated ids in the response are dropped with a warning; a response that
  keeps nothing at the top level triggers keep-all + warning (never silently
  return an empty corpus).

### 2.2 Prompt

`CHILD_SELECT_PROMPT` — recall-biased but comparative: "select every child
that could plausibly contain an answer; if all are equally unlikely, select
none." Versioned via `PROMPT_VERSIONS` so the debug cache and MCQ precompute
invalidate correctly.

### Tests — Phase 2
`tests/test_batched_prune.py`
- N siblings → exactly 1 call (not N); prompt contains each child's summary
  and no grandchild content
- pruned section marks every descendant pruned, progress counter still
  reaches total (regression guard on `:943-962`)
- unknown id in response ignored; empty top-level keep → keep-all + warning
- chunking path: 50 children with a small budget → k calls, union correct,
  `batched` flag set
- malformed JSON → retry → `error` status, subtree kept (conservative)

---

## Phase 3 — BM25 queue + context-budgeted leaf evaluation

### 3.1 Ranking

`ranking.py` — self-contained Okapi BM25 (~60 lines, no new dependency;
`requirements.txt` currently holds only `ollama`).

- Corpus = the surviving candidate leaves for this run (across all docs, so
  ranking is global rather than per-document).
- Query = the user's query. Tokenisation: lowercase, strip punctuation,
  simple stopword list. Deterministic and testable.
- Ties broken by document order so runs are reproducible.

### 3.2 Budgeted evaluation loop

**Confirmed semantics:** the cap is on **accepted** nodes, measured against
the **agent's** context — that is where every retrieved node ends up for
reasoning. Each leaf eval is its own call, so evaluation context is not
cumulative; only accepted content accumulates against `agent_ctx`.

```
queue = leaves sorted by BM25 desc
budget = runtime.agent_ctx - reserve - prompt_overhead - estimate(query)
for leaf in queue:                     # dispatched through WORK_POOL, in order
    if accepted_tokens + estimate(leaf) > budget: mark rest DEFERRED; break
    verdict = evaluate_leaf(leaf)
    if verdict.relevant: accepted_tokens += estimate(leaf)
```

- Rejected leaves cost no budget, so we keep walking the queue — the loop is
  bounded by the budget in *accepted* content, giving time linear in the
  context window as intended.
- A second cap `max_leaf_evals` (config, default generous) bounds wall-clock
  independently; hitting it also produces `deferred`, with a distinct reason.
- Parallel dispatch stays in-order-committed: results are applied in queue
  order so the budget decision is deterministic regardless of completion race.

### 3.3 `deferred` is a first-class status

Distinct from `pruned` (LLM decided against) and `rejected` (evaluated, not
relevant). `deferred` = *never evaluated, budget exhausted*.

- New status in `RunContext.events` + `node_meta`.
- Its own colour and legend entry in the treemap and graph/boxes views
  (`ui/index.html:240-243, 286-293`, `ui/main.js` verdict colour maps).
- `/api/run` and `/api/chat` return:
  `budget: {evaluated, deferred, accepted_tokens, budget_tokens, capped_by}`
  and the deferred node list with BM25 scores.
- Chat UI: a banner under the answer — *"N further passages ranked below the
  context budget were not read. Browse them →"* — linking into the retrieval
  tab filtered to deferred nodes. The user can inspect and manually promote
  one into a follow-up.

### 3.4 Leaf evaluator prompt

Slight rewrite: it now receives the parent's **real summary** (Phase 1) in
place of today's `"Covers: A, B, and 3 more"` string (`pageindex.py:1034`),
plus its BM25 rank as weak context.

### Tests — Phase 3
`tests/test_ranking.py`, `tests/test_budget_eval.py`
- BM25 correctness on a fixture corpus; determinism across runs; tie-break
- budget loop: everything fits → zero deferred; nothing fits → all deferred,
  no crash, empty-answer path is graceful
- rejected leaves don't consume budget (walk continues past them)
- `max_leaf_evals` cap produces `deferred` with `capped_by: "max_evals"`
- out-of-order thread completion yields identical deferred set (determinism)
- `deferred ∩ pruned = ∅`, `evaluated + deferred + pruned == total leaves`
  (invariant test — the accounting bug class this design is most prone to)

---

## Phase 4 — Ingest rework + folder tags

### 4.1 Recursive discovery

`betteringest_pdf.run:188` and `run_paths:112` use `glob("*.pdf")` →
`rglob`. `basic_markdown.run` likewise for `*.md`.
`/api/ingest/source:475` currently rejects a folder whose PDFs are all in
subfolders — fix with the same change.

### 4.2 Doc identity (prerequisite, not optional)

Recursion makes stem collisions real: `internal/report.pdf` and
`arxiv/report.pdf` both write `knowledge_base/report.md` today.

- `doc_id` = sanitised relative path, separators → `__`
  (`arxiv__2401_12345`), unique by construction.
- `title` keeps the original human-readable stem for display.
- All flat paths key on `doc_id`: `knowledge_base/<doc_id>.md`,
  `index/<doc_id>.json`, `knowledge_base/assets/<doc_id>/`.
- Residual collisions (different paths sanitising alike) → numeric suffix +
  warning.

### 4.3 Central manifest

`modules/ingest/_manifest.py`, written by **both** ingest modules (today only
`betteringest_pdf` writes `.sources.json`, so markdown docs have no metadata).

`knowledge_base/.manifest.json`:
```json
{"arxiv__2401_12345": {
   "title": "...", "source_path": "...", "rel_path": "arxiv/2401_12345.pdf",
   "sha256": "...", "tags": ["arxiv"], "manual_tags": [],
   "ingest_module": "...", "ingested_at": "..." }}
```

- **Auto tags** = every path component of the relative parent directory
  (`papers/arxiv/2024/x.pdf` → `["papers", "arxiv", "2024"]`). Nested folders
  yield nested tags; that's the intended "type of source" signal.
- `manual_tags` is a separate field so a re-ingest never clobbers hand
  curation. Effective tags = `auto ∪ manual`.
- One-time migration reads existing `.sources.json`; the old file keeps being
  written for a release so nothing breaks mid-upgrade.
- `sha256` is what Phase 5 keys its persistent per-leaf precompute on.

### 4.4 Tags stay out of `index/*.json`

Doc-level metadata joined in at retrieval time — re-tagging must not require
re-indexing. `_run_retrieval` attaches `tags` to each doc's result and to each
source in the `/api/chat` payload.

### 4.5 API surface (backend only — frontend wiring is later, per your note)

- `GET /api/tags` → `[{tag, doc_count}]`
- `GET /api/documents` → each doc gains `tags`, `title`, `doc_id`
- `POST /api/run` / `POST /api/chat` accept optional `tags: [...]`, filtering
  documents **before** the retrieval loop (OR within the list; empty = all).
  This is the cheapest possible win against corpus-size scaling.

### Tests — Phase 4
`tests/test_manifest_tags.py`, extend `tests/test_betteringest_ingest.py`
- nested tmp tree → correct `doc_id`s, correct auto tags per depth
- same-stem-different-folder → two distinct docs, no overwrite (the bug this
  phase exists to prevent)
- sanitisation collision → suffix + warning
- `manual_tags` survive a re-ingest; effective-tag union correct
- `.sources.json` → `.manifest.json` migration is lossless
- tag filter in `_run_retrieval` skips non-matching docs with **zero** LLM
  calls (asserted on the fake client's call count)

---

## Phase 5 — MCQ precompute (per-leaf, persistent, OR)

Per your decision: every leaf is judged against every MCQ answer at index
time; the query-time subtree is the **union** of the selected answers' leaf
sets.

### 5.1 Question config

`config/mcq.json` — versioned, placeholder questions for now:
```json
{"version": 1,
 "questions": [{"id": "q1", "prompt": "...", "multi_select": true,
                "answers": [{"id": "q1a1", "label": "...",
                             "facet": "<what the LLM judges leaves against>"}]}]}
```
`facet` is deliberately separate from `label`: the user-facing wording and the
LLM-facing criterion should be tunable independently.

### 5.2 Precompute

- **One call per (leaf, question)**, not per (leaf, answer): the prompt lists
  that question's answers and returns the subset the leaf is relevant to.
  Cost drops from `leaves × answers` to `leaves × questions`.
- `FACET_LEAF_PROMPT` — deliberately **recall-biased**. A false negative here
  is unrecoverable (the leaf can never reach the user's query), a false
  positive only costs a little budget in Phase 3. Prompt says so explicitly.
- Persisted at `index/.facets/<doc_id>.json`, keyed
  `leaf_content_sha256 → {question_id → [answer_ids], model, prompt_version}`.
  Content-hash keyed, so it survives re-index and re-ingest of *unchanged*
  documents, and recomputes only for changed leaves — exactly your requirement.
- Runs as a background job (own `RunContext`, so it cannot corrupt a live
  query's treemap — this is why Phase 0.3 comes first), with
  `GET /api/mcq/precompute/progress` and a resume-on-restart path.

### 5.3 Query-time assembly

1. Selected answers → union of their leaf sets (OR, per your call).
2. Ancestor closure of that leaf set → a well-formed connected subtree.
   (Operating on leaves and deriving internals — rather than intersecting
   internal-node sets — is what keeps the tree connected.)
3. That subtree is the input to Phase 2 pruning, then Phase 3's queue.
4. Empty selection → whole corpus. Union empty → fall back to full tree with
   an explicit notice, never a silent empty result.

### 5.4 Tags from MCQ

Each surviving leaf carries `mcq_tags: ["q1a2", "q3a1"]` — which answers put
it in the candidate set. Under OR this is genuinely discriminating (leaves
differ in membership), so it flows into the node payload and the source cards
alongside the folder tags from Phase 4, sharing one tag surface:
`node_tags[node_id] = {"source": [...], "mcq": [...]}`.

### 5.5 Visibility on set growth

Because selections union, the UI reports candidate-set size as answers are
picked (`k leaves selected of N`). Not a design change — just making the
widening observable rather than surprising.

### Tests — Phase 5
`tests/test_mcq_precompute.py`
- one call per (leaf, question), never per answer; count asserted
- unchanged doc re-index → zero calls; one edited leaf → exactly one call
  per question for that leaf
- prompt-version / model bump invalidates the whole cache
- union semantics: two answers → union, order-independent, idempotent
- ancestor closure is connected and contains every selected leaf
- empty selection → full corpus; empty union → fallback + notice
- precompute concurrent with a live retrieval → neither run's events polluted
  (the Phase 0.3 payoff, asserted directly)

---

## Phase 6 — Debug cache

Debug-only, as agreed: no similarity-based production cache.

### 6.1 Key + storage

`.debug_cache/<sha256>.json`, keyed on:
`normalised query + index fingerprint + retrieval model + synthesis model +
prompt versions + mcq selection + tag filter`.

Index fingerprint = hash over every `index/*.json` content hash — the thing
that makes a stale replay impossible after re-ingest.

Stored payload: the **full** `_run_retrieval` results (`node_meta`, verdicts,
reasons, quotes), the live-events snapshot, the budget block, BM25 ordering,
and the synthesis answer separately.

### 6.2 Behaviour (your option b + badge)

- `GET /api/cache/lookup?query=…&…` → `{hit, key, cached_at, node_count}`
- Frontend, on submit, checks first: **"You asked this before (2h ago, 41
  nodes). Replay the cached run, or re-run live?"**
- `GET /api/cache/questions` → cached questions for **autocomplete**,
  rendered through the existing suggestions box
  (`ui/main.js:435 renderSuggestions`, `ui/index.html:318`) as a distinct
  "Previously asked" group alongside the test cases.
- Replay **restores the live-events snapshot** into a fresh `RunContext`, so
  the treemap renders fully instead of showing an empty visualisation —
  the trap here is that `start_run` clears all event sets.
- **CACHED badge** on the chat answer and in the retrieval tab header, with
  the cache timestamp. Unmissable.
- Synthesis is re-run live by default (so you can iterate on the synthesis
  prompt against a frozen node set); a checkbox replays the cached answer too.
- `POST /api/cache/clear`; whole feature behind `debug_cache_enabled`
  (default off in `/api/config`).

### Tests — Phase 6
`tests/test_debug_cache.py`
- key changes when *any* component changes; identical inputs → hit
- re-index (index fingerprint change) → miss, not a stale replay
- round-trip fidelity: replayed results deep-equal the live results
- replay repopulates live events (treemap-not-empty regression guard)
- disabled flag → always miss, nothing written
- autocomplete lists only cached questions valid for the current fingerprint

---

## Test infrastructure

Verbose terminal output, as requested.

### Configuration
`pytest.ini`:
```ini
[pytest]
addopts = -v -ra --tb=short --color=yes --durations=10
markers =
    unit: offline, deterministic (default)
    llm: requires a live Ollama (deselected unless -m llm)
```

### `tests/conftest.py`
- `fake_ollama` fixture: a recording client monkeypatched over
  `pageindex._chat` / the client factory. Records every prompt, returns
  scripted responses. **Every logic test is offline and deterministic** —
  no phase's correctness depends on model behaviour.
- `tmp_corpus` fixture: builds a nested tmp docs tree + KB + index.
- `assert_call_count` helper — several phases' whole point is *fewer calls*,
  so call counts are asserted, not assumed.
- A `pytest_terminal_summary` hook printing a per-area table:

```
─────────── ASEPSIS RETRIEVAL SUITE ───────────
  Phase 0 · foundations      14 passed
  Phase 1 · summaries        11 passed   1 failed
  Phase 2 · batched prune     9 passed
  Phase 3 · budget + BM25    16 passed   2 failed
  Phase 4 · tags + ingest    12 passed
  Phase 5 · mcq precompute   10 passed
  Phase 6 · debug cache       8 passed
  ── FAILURES ──
  Phase 1 · test_summaries.py::test_incremental_rebuild
  Phase 3 · test_budget_eval.py::test_deferred_accounting
```

### Runner
`run_tests.py` (and a `make test` target): runs the offline suite by default,
`--llm` adds the live-Ollama tier, `--phase N` filters to one phase.

### Live tier (`-m llm`)
Small set, opt-in, real Ollama: summaries are non-empty and shorter than
their input; batched prune returns parseable JSON with valid ids; end-to-end
retrieval on the fixture corpus still passes the existing `TEST_CASES`
(`server.py:206`) — the regression net for the whole rework.

---

## Resolved decisions

1. **Model.** `gemma4:e4b`, `max_ctx` 128k, `default_ctx` 48k, frontend
   slider over `[4096, 128k]` with the 32–64k band marked recommended.
   Same model for retrieval and synthesis unless overridden. → Phase 0.1
2. **Budget semantics.** Cap is on **accepted** nodes against the **agent's**
   context, since that is where all retrieved nodes land. Rejected leaves
   cost nothing and the queue walk continues past them. → Phase 3.2
3. **Index migration.** Forced full re-index on first run of the new code
   (existing `index/*.json` have no summaries or hashes). Detected via an
   `indexFormatVersion` field: missing/older → rebuild all, with a blocking
   progress screen and a clear explanatory message. Now a build step, not an
   open question. → Phase 1.5 below.

## Remaining open question

**Multiple-choice question content and schema (Phase 5 only).** "MCQ" was just
shorthand for the multiple-choice questions from your original item 2 — the
placeholder questions shown before the user types, whose answers pre-select a
subtree. Phase 5 ships `config/mcq.json` with 3 placeholder questions in the
shape given above, including the `label` (shown to the user) vs `facet` (what
the LLM judges leaves against) split. Nothing else blocks on it; review the
placeholders when Phase 5 lands and replace the wording then.

---

## Phase 1.5 — Forced re-index migration

- `INDEX_FORMAT_VERSION` constant written into every `index/<doc>.json`.
- On server start and before any retrieval, any index missing the field or
  carrying an older version marks the corpus stale.
- Stale corpus → `/api/status` reports `index_stale`, the UI shows a blocking
  "Rebuilding index (new format: summaries + hashes)" screen with per-doc
  progress, and retrieval endpoints return 409 until it completes.
- The rebuild is the ordinary `build_index` path, so it is incremental from
  then on — this cost is paid exactly once.
- Test: an old-format index triggers exactly one full rebuild; a second start
  triggers none.
