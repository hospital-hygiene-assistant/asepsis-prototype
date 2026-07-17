# ASEPSIS backend and retrieval console

This repository provides the local FastAPI backend and a vanilla-JavaScript
**retrieval debug console**. The German Next.js application in the sibling
`frontend/` repository is the only practitioner surface. The console described
below exists for corpus, retrieval, and provenance inspection; its chat is not a
second product surface.

ASEPSIS is a retrieval-augmented-generation (RAG) **exploration and explainability** tool. It
indexes a corpus of heading-structured documents (currently medical guidelines),
answers a natural-language query by walking the document trees with an LLM, and
visualizes **which parts of each document the model selected, rejected, or skipped —
and why**.

The debug console has **two tabs that share one backend run state**:

- **💬 Chatbot** — a diagnostic mirror of the answering workflow. A question triggers the exact same retrieval workflow
  as the Retrieval tab, then a synthesis call produces a structured, citation-anchored
  answer (`Short answer · Recommended action · Rationale · Limitations`) with a
  grounding badge. Each inline `[n]` citation links to a source card showing the
  passage with its **deciding quote highlighted**, the model's *why selected* reason,
  the **original PDF page with the passage's bounding box highlighted** (for
  PDF-ingested documents), a live **retrieval-reasoning trace** (every kept / pruned /
  rejected / retrieved decision as it happens), and buttons into the annotated reader
  and the Retrieval tab. Verified sources stay inline with the answer they support.
- **⊟ Retrieval** — the original explorer: Library → Ask → Review workflow, live
  treemap, graph/boxes verdict views, snippets, reader, and "why not?" explanations.

Asking in the Chatbot animates the Retrieval tab's treemap live, and the finished
run is immediately inspectable there — the two tabs are two lenses on one run.

**One-click start (macOS):** install Python 3.12, `uv`, and Ollama, then
double-click **`run_tauri.command`**. It reproduces the locked Python environment,
starts Ollama with GPU visibility disabled when needed, pulls the configured model
if needed, and launches the app. PDF OCR/table review dependencies are installed
separately with `uv sync --frozen --group ocr` because Paddle is large.

This README walks through a full session as a **timeline**. At each timestamp `t`
the same moment is described from two angles:

- 🖥️ **Frontend view** — what the user sees, what actions are available, what each does.
- ⚙️ **Backend view** — what the server/LLM is doing, and what it produces.

---

## Table of contents

- [Two views, one timeline](#two-views-one-timeline)
- [System at a glance](#system-at-a-glance)
- [Timeline](#timeline)
  - [t0 — Launch](#t0--launch)
  - [t1 — App ready (welcome screen)](#t1--app-ready-welcome-screen)
  - [t2 — Browsing the corpus (before any query)](#t2--browsing-the-corpus-before-any-query)
  - [PDF ingest review](#pdf-ingest-review)
  - [t3 — Asking a question](#t3--asking-a-question)
  - [t4 — Retrieval in progress](#t4--retrieval-in-progress)
  - [t5 — Results arrive](#t5--results-arrive)
  - [t6 — Inspecting the answer](#t6--inspecting-the-answer)
  - [t7 — Asking "why not?" (on-demand)](#t7--asking-why-not-on-demand)
  - [t8 — Iterating](#t8--iterating)
- [Reference](#reference)
  - [Keyboard shortcuts](#keyboard-shortcuts)
  - [HTTP API](#http-api)
  - [Ingest adapters](#ingest-adapters)
  - [Ollama instances](#ollama-instances)
  - [Verdict color scheme](#verdict-color-scheme)
  - [Project layout](#project-layout)
- [Running the tests](#running-the-tests)

---

## Two views, one timeline

The application is a **Tauri desktop window** (a WKWebView) that loads a UI served by a
**local FastAPI backend** on `http://127.0.0.1:8765`. The same UI runs in a normal
browser if Tauri isn't installed.

```
┌─────────────────────────────┐        HTTP (localhost:8765)        ┌──────────────────────────────┐
│        FRONTEND              │  ───────  GET / , /api/* ────────▶  │          BACKEND               │
│  Tauri WKWebView / browser   │                                     │  FastAPI (tauri-app/server.py) │
│  index.html · main.js · D3   │  ◀──────  JSON + live status ─────  │  pageindex/ (index + LLM)      │
└─────────────────────────────┘                                     │  Ollama (Gemma 4 E2B Q4) pool  │
                                                                     └──────────────────────────────┘
```

The frontend never talks to Ollama directly — it only calls the FastAPI endpoints. The
backend owns the document index, the LLM calls, and all run state.

---

## System at a glance

| Layer | Tech | Where |
|-------|------|-------|
| Desktop shell | Tauri (WKWebView) | `tauri-app/src-tauri/` |
| UI | HTML + vanilla JS + D3 v7 (vendored offline) | `tauri-app/ui/` |
| Server | FastAPI + Uvicorn, port `8765` | `tauri-app/server.py` |
| Index + retrieval | Deterministic heading parser + LLM retrieval | `pageindex/` |
| LLM runtime | Ollama, model `gemma4:e2b-it-q4_K_M` | pool on `11434+`, explainer on `11500` |
| Ingest adapters | alternate source-to-markdown producers | `modules/ingest/` |

Retrieval is **two-phase**: a cheap top-down **section-pruning** pass (BFS over the
heading tree) followed by a focused **per-leaf evaluation** of every surviving leaf.
Both phases run in parallel across the configured Ollama instances.

---

## Timeline

### t0 — Launch

🖥️ **Frontend**
- **Action available:** run `uv run --frozen python run-tauri.py` (optionally
  `--ollama-instances N`).
- **Outcome:** a desktop window titled *Asepsis Prototype* opens (or the system browser, if Tauri's CLI isn't installed). Nothing is interactive yet — the UI is loading.

⚙️ **Backend**
- `run-tauri.py` verifies its dependencies and opens the current valid Expected
  library. It never replaces an operator-reviewed generation from stale staging
  files. Set `ASEPSIS_REBUILD_LIBRARY=1` only for an intentional one-time rebuild
  from `docs/` and `knowledge_base/`.
- Starts FastAPI on `127.0.0.1:8765` and waits until it responds, then opens the window pointing at it.
- **Outcome:** the server is up; one immutable Expected library generation binds every document tree to its canonical Markdown and available source evidence. No LLM work yet.
- *Note:* every response is sent `Cache-Control: no-store`, and asset URLs are versioned by file mtime, so the webview never serves a stale UI across launches.

---

### t1 — App ready (welcome screen)

🖥️ **Frontend**
- On load the UI fires five calls in parallel (`/api/tests`, `/api/modules`, `/api/config`, `/api/status`, `/api/documents`) and renders:
  - **Sidebar:** the ingest-adapter dropdown, an Ollama-instance stepper with live activity dots, and the test-case list grouped into *Single document*, *Multi-section*, *Cross-document*.
  - **Main area:** a **treemap of the whole corpus** — every document is a box subdivided into its sections and leaves. All boxes start neutral ("pending").
  - **Chat launcher:** a collapsed *“💬 Ask a question ⌘K”* pill at the bottom-right.
- **Actions available:** browse documents, pick a test, open the chat, change the ingest adapter, or change the instance count.

⚙️ **Backend**
- Serves the cached index trees (`/api/documents`) and the static test definitions (`/api/tests`). No LLM activity; instances report **idle** via `/api/status`.

---

### t2 — Browsing the corpus (before any query)

🖥️ **Frontend**
- **Action:** click any box in the welcome treemap.
- **Outcome:** the **document viewer modal** opens with the rendered markdown of that document, a clickable table of contents on the left, and click-to-scroll to a section. Because no query has run yet, this is a **plain reading view** — no verdict colors.

⚙️ **Backend**
- The `/api/documents` listing supplies a generation-scoped `full_href`; it returns canonical Markdown from the same immutable generation as the displayed tree. The console renders it with the vendored `marked.js` and derives the TOC from the headings.

---

### PDF ingest review

🖥️ **Operator console**

- Select the BetterIngest PDF adapter and a local PDF or folder. OCR proposes
  figure and table rectangles, then the ingest pauses in a modal review.
- Add, move, resize, or delete rectangles. Mark new rectangles as figures or
  tables. Undo and redo are local to the annotation tool.
- Every table must either produce focused structured recognition or have its
  recognition failure explicitly acknowledged. **Publish** is disabled until
  the complete review is ready.

⚙️ **Backend**

- The review is persisted for 24 hours with an opaque identity and revision.
  A stale browser cannot overwrite a newer edit, and HTTP responses expose no
  source filesystem paths.
- Confirmation publishes the entire reviewed batch through the immutable
  Expected library interface. If validation fails, the previous generation
  remains current.

---

### t3 — Asking a question

🖥️ **Frontend** — two ways to start a run:
- **Pick a test case** (sidebar) — fills the query and runs immediately.
- **Type a custom query** in the chat:
  - **Open the chat** with `⌘K` / `Ctrl+K`, the `/` key, or by clicking the pill. It expands and focuses instantly.
  - The input **auto-grows** as you type, up to ~10 lines, then scrolls. `Esc` (or the ⌄ button) collapses it back to the pill to maximize the visualization area.
  - **Submit** with `Enter` (or the send button). `Shift+Enter` inserts a newline.
- Optionally first change the **ingest adapter** or the **Ollama instance count** (the stepper POSTs to the backend and the activity dots reflect the new pool).

⚙️ **Backend**
- The frontend `POST`s to `/api/run` with `{query | test_id}`; retrieval always uses the one PageIndex implementation.
- The server counts total leaves across all documents and calls `start_run(total)` to reset the live counters, then begins retrieval per document.

---

### t4 — Retrieval in progress

🖥️ **Frontend**
- The view switches to the **loading state**: the corpus treemap stays on screen and **animates live** — boxes turn green (retrieved), rose (rejected), or grey (pruned) as decisions land. A counter shows `Evaluating X / Y leaves`, and the sidebar instance dots spin while busy.
- **Action available mid-run:** click a box to open the **document viewer while the run is happening**. The markdown is annotated **in real time** — sections gain their verdict color the moment the model decides, and **hovering a paragraph shows the model's live decision** (status + reason), marked `· live`.
- The frontend polls `/api/status` every **250 ms** to drive all of the above.

⚙️ **Backend** — two-phase retrieval (`pageindex.search_document`):
1. **Top-down pruning** (`_check_section_relevant`): BFS over the heading tree; each section is judged (from its full descendant outline) as possibly-relevant or not. Pruned branches are skipped entirely — their leaves are counted as "done" but never read. *Errs toward inclusion to avoid false negatives.*
2. **Per-leaf evaluation** (`_evaluate_leaf`): every surviving leaf gets one focused LLM call returning `{relevant, reason, quote}`; the `quote` must be verbatim from the content.
- Both phases run via a `ThreadPoolExecutor`, **round-robin across the Ollama pool** (each leaf is assigned to exactly one instance — no duplicated work).
- As each verdict is made it is published to `/api/status` under `live` (`retrieved / rejected / kept / pruned` id sets) and `live.meta` (`{node_id: {status, reason, quote}}`), which is what powers the live treemap and live doc viewer.

---

### t5 — Results arrive

🖥️ **Frontend** — `/api/run` returns and the **results view** replaces the welcome screen:
- **Status bar:** the query, the pipeline chips (① ingest · ② index · ③ query), and a **PASS / FAIL** badge (only for test cases; custom queries show *Custom Query*).
- **Document cards:** one per document with a retrieval ratio (`retrieved / total leaves`). Clicking a card switches the active document; each card also has a **⤢ Read** button.
- **View tabs:** **⊟ Graph** (a D3 hierarchy tree of the active document) and **▦ Boxes** (a treemap of the active document), both colored by the final verdicts.
- **Snippets panel:** every retrieved leaf as a card — document/section tags, the model's reason, and the full content with the **deciding quote highlighted**.

⚙️ **Backend**
- The `/api/run` debug adapter derives, per document, the raw tree, `retrieved_ids`, `node_meta` (`{status, reason, quote}` per node), and retrieved leaf content from the typed search result and exact immutable generation. For test cases it also returns the pass/fail evaluation (`missing` / `any_missing` against the test's `expected` / `expected_any`).

---

### t6 — Inspecting the answer

🖥️ **Frontend** — ways to understand *why*:
- **Hover any node** in the Graph tree → a tooltip with its verdict (Retrieved / Kept / Pruned / Rejected) and the model's reason; retrieved leaves also show the quote.
- **Hover a tree edge** → a **path-reasoning** tooltip: the root→leaf chain with each step's verdict and reason (the audit trail).
- **Click ⤢ Read** on a document card → the **document viewer with verdict overlays**:
  - Each heading + its paragraphs is tinted by verdict (sage / rose / blue / grey), the **deciding quote is highlighted in amber**, and the **TOC entries mirror the same colors**.
  - **Hover a paragraph** → the model's decision for that section (no `· live` marker now — these are the final results).
- **Boxes tab** → the same verdicts as a treemap; clicking a box opens the reader scrolled to that section.
- **Graph controls** (top-right): adjust card size, node spacing, and hide/show the snippets panel.

⚙️ **Backend**
- No second retrieval path — everything here is rendered by the debug adapter from typed passage decisions and generation-scoped canonical Markdown.

---

### t7 — Asking "why not?" (on-demand)

🖥️ **Frontend**
- **Action:** in the document viewer, **click a rejected or pruned passage**.
- **Outcome:** a **“Why not selected”** popover appears (spinner → result). The model re-reads the section and returns a **grounded topic** ("what this section actually covers") plus a one-line reason contrasting that topic with the query. If, on re-read, it now thinks the section *does* answer the query, the popover flags it for a manual check.
- Results are **cached per node** — re-clicking is instant, and subsequent hovers show the cached explanation inline.

⚙️ **Backend**
- `POST /api/explain {stem, node_id, query}` → `explain_nonselection` runs a content-anchored prompt.
- It runs on a **dedicated Ollama instance on port 11500**, started **lazily on first use** and kept warm afterward, so explanations never steal a slot from an in-flight query.
- *Design note:* a positive verdict is anchored by a verbatim quote; a negative has no such anchor and is easy to confabulate. So instead of asking "why did you reject this", the prompt asks the model to **describe the section's actual topic** (verifiable against the text) and keeps a strict relevant/not-relevant decision.

---

### t8 — Iterating

🖥️ **Frontend**
- **Run another query** (chat or another test) — the loading/results cycle repeats.
- **Switch documents** via the cards; **toggle Graph/Boxes**; **scale the Ollama pool** with the stepper (more instances = more parallelism on the next run).
- **Re-open the reader** any time to revisit verdicts and explanations.

⚙️ **Backend**
- Each run resets the live counters (`start_run`) and recomputes verdicts. Adjusting the instance count `POST`s `/api/config`, which starts/stops `ollama serve` subprocesses and reconfigures the client pool.

---

## Reference

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `⌘K` / `Ctrl+K` | Toggle the chat (open + focus, or collapse) |
| `/` | Open the chat (when not already typing) |
| `Enter` | Run the query |
| `Shift+Enter` | Newline in the query |
| `Esc` | Collapse the chat / close the document viewer / dismiss the explain popover |

### HTTP API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | The UI (asset URLs versioned by mtime) |
| `GET` | `/api/tests` | Built-in test cases |
| `GET` | `/api/config` · `POST` `/api/config` | Read / set the Ollama instance count |
| `GET` | `/api/modules` | Available ingest adapters |
| `GET` | `/api/documents` | All indexed document trees |
| `GET` | `/api/library/{generation}/documents/{document}/full` | Canonical Markdown from one immutable generation |
| `GET` | `/api/library/{generation}/documents/{document}/pdf` | Source PDF bound to that generation |
| `GET` | `/api/library/{generation}/documents/{document}/assets/{asset}` | Provenance asset bound to that generation |
| `GET` | `/api/status` | Ollama instance activity |
| `POST` | `/api/ingest/run` · `/api/ingest/add` | Prepare an ingest batch; review-required adapters stop before publication |
| `GET` | `/api/ingest/reviews/{review}` | Resume one revisioned operator review |
| `PUT` | `/api/ingest/reviews/{review}/documents/{document}/regions` | Replace reviewed geometry at an expected revision |
| `POST` | `/api/ingest/reviews/{review}/documents/{document}/regions/{region}/recognize-table` | Recognize one confirmed table crop |
| `POST` | `/api/ingest/reviews/{review}/confirm` | Atomically publish one ready review |
| `GET` | `/api/runs/{run_id}` | Progress and verdict state for one retrieval run |
| `POST` | `/api/run` | Run retrieval for a query/test |
| `POST` | `/api/chat` | Versioned grounded answer, search-coverage facts, and verified sources |
| `POST` | `/api/explain` | On-demand grounded "why not selected" (dedicated instance) |

### Ingest adapters

The two real ingest alternatives are discovered from `modules/ingest/*.py`.
Indexing, whole-library retrieval, and question answering each have one direct
implementation; they are not configurable plugin stages.

The former `index/` layout is not read. After upgrading an existing checkout,
run `uv run python pipeline.py ingest` and `uv run python pipeline.py index` to publish the
first Expected library generation. If that checkout has a pre-version-2
`knowledge_base/.sources.json`, remove that staging manifest before ingest; a
legacy manifest is rejected rather than trusted as source evidence.

| Adapter | Role |
|---------|------|
| `basic_markdown` | Copy prepared `docs/*.md` into staging for direct publication |
| `betteringest_pdf` | Reconstruct PDFs, then require operator review of every proposed figure and table before publication |

### Ollama instances

- **Retrieval pool:** ports `11434, 11435, …`, sized by the sidebar stepper / `--ollama-instances`. Used round-robin for the two-phase retrieval.
- **Explainer:** port `11500`, dedicated to `/api/explain`, started lazily and kept warm — isolated so on-demand explanations never disturb a running query.
- Model: `gemma4:e2b-it-q4_K_M` (Gemma 4 E2B instruction model, Q4_K_M).
- Processes started by ASEPSIS request CPU-only execution by default. `/api/config`
  reports whether loaded models actually have zero VRAM allocation. A separately
  started Ollama process remains outside ASEPSIS control. See
  [`docs/runtime-models.md`](docs/runtime-models.md).

### Verdict color scheme

One muted palette is shared across the treemap, the Graph tree, the snippet highlights, the document-viewer bands, and the TOC:

| Verdict | Color | Meaning |
|---------|-------|---------|
| **Retrieved** | sage green | Leaf the model judged directly answers the query (has a quote) |
| **Rejected** | dusty rose | Leaf the model read and judged not relevant |
| **Kept** | blue | Section that passed pruning (its children were explored) |
| **Pruned** | grey | Branch skipped without reading (its parent section was pruned) |
| **Deciding quote** | amber | The verbatim sentence(s) that justified a retrieval |

### Project layout

```
astepsis/
├── docs/                     # Source documents (.md)
├── knowledge_base/           # Ingested markdown  (ingest output)
├── library/                  # Immutable indexes, Markdown, source bindings, and content-addressed objects
├── .ingest_reviews/          # Private, expiring operator review state (ignored)
├── paths.py                  # Where the data lives — one definition, imported everywhere
├── pageindex/                # The retrieval engine
│   ├── nodes.py · build.py   #   heading tree; writing it (deterministic, no model)
│   ├── search.py · llm.py    #   pruning and judging (model-guided); the model call
│   ├── prompts.py · pins.py  #   retrieval prompts; provenance blocks from PDF ingest
│   └── settings.py · clients.py · document_index.py · question_run.py
├── retrieval_cases.py        # Retrieval quality cases — shared by the suite and the console
├── pipeline.py               # CLI: ingest → immutable index generation → grounded query
├── run-tauri.py              # One-command launcher (pipeline + server + window)
├── run_tauri.command         # Double-clickable macOS launcher (installs deps, starts Ollama)
├── modules/                  # ingest adapters; review lifecycle and focused table recognition
├── tauri-app/
│   ├── server.py             # Entrypoint: assembles the app
│   ├── api/                  # One router per concern, plus what no single route owns
│   ├── src-tauri/            # Tauri desktop shell config
│   └── ui/                   # The dev console (not the practitioner surface)
└── tests/                    # Deterministic; tests/test_retrieval.py needs Ollama
```

## Running the tests

```bash
uv sync --frozen
uv run ruff check . --select F,E9
uv run pytest -q --cov --cov-report=term --cov-fail-under=80
uv run pytest tests/test_retrieval.py -v  # live quality cases; requires Ollama
```

The deterministic gate installs Chromium with `uv run playwright install
chromium`. Install the optional local PDF models with `uv sync --frozen --group
ocr`; CI exercises their adapters with fakes and does not download Paddle
weights.

The v3 chat examples are generated by the backend and consumed in both
repositories. Verify that neither copy has drifted from the authoritative
producer before handoff:

```bash
PYTHONPATH=.:tauri-app uv run python -m api.chat_contract_fixtures --check \
  tests/contracts/chat_v3 \
  ../frontend/src/lib/api/__fixtures__/chat_v3
```

Use `--write` instead of `--check` only when intentionally regenerating both
copies after a wire-format change.

## Deployment scope

This prototype is a local tool: the backend binds to loopback and trusts the
person operating that machine. Its folder-path ingest controls and model-pool
controls are deliberately not exposed as a hosted product interface. Before
any network deployment, authenticated access, safe file uploads, bounded job
orchestration, and an operational security review are required. See ADR 0002.
