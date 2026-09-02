# Asepsis — User Guide

Asepsis answers clinical questions from **your own documents**, and shows you
exactly where every part of the answer came from.

It is not a search engine and not a general chatbot. It reads the guidelines,
protocols and papers you have given it, finds the passages that bear on your
question, and writes an answer in which **every claim points back at the passage
it came from** — so you can check it rather than trust it.

Everything happens on this machine. Nothing is uploaded.

---

## Contents

- [Before you start](#before-you-start)
- [The first two minutes](#the-first-two-minutes)
- [Asking a question](#asking-a-question)
- [Reading the answer](#reading-the-answer)
- [Following up](#following-up)
- [When the answer falls short](#when-the-answer-falls-short)
- [Looking behind the answer](#looking-behind-the-answer)
- [Adding your own documents](#adding-your-own-documents)
- [Settings worth knowing about](#settings-worth-knowing-about)
- [Keyboard shortcuts](#keyboard-shortcuts)
- [Common questions](#common-questions)

---

## Before you start

On macOS, **double-click `run_tauri.command`**. It does the setup for you:
creates a private Python environment, installs the app's dependencies, starts
Ollama if it is not already running (tuned for parallel retrieval), and
downloads the language model on first launch.

Two things it cannot install for you, and will tell you about with a link if
they are missing:

- **Python 3** (3.11 or newer)
- **Ollama** — install it and open it once

You need a network connection the **first** time, for the dependency install
and the model download. After that the app runs entirely offline.

If the Tauri desktop tooling is present the app opens in its own window;
otherwise it opens in your browser. Both are the same app.

---

## The first two minutes

Open the app. A **guided tour** appears the first time — eight cards showing
what each part does, on example data. Nothing in it runs; it is a walk-through,
not a demo you have to sit through. Close it whenever you like.

You can reopen it any time: **⚙ Settings → Take the guided tour**.

There are two tabs along the top:

| Tab | What it is for |
|---|---|
| **💬 Chatbot** | Ask questions, read answers, check sources. This is where you will spend your time. |
| **⊟ Retrieval** | The working behind an answer — every section the app considered, and what it decided about each. |

They are two views of the same thing. Asking a question in the Chatbot fills the
Retrieval tab in as it goes.

---

## Asking a question

Type it the way you would ask a colleague. Full sentences help — *"what empiric
antibiotics for abdominal septic shock?"* works better than *"septic shock abx"*,
because the app is matching meaning rather than keywords.

**While it works**, the passages it has already found appear as a deck of cards
you can read through. This is not a loading animation — those are the actual
passages your answer is about to be written from, and reading them while you wait
costs you nothing.

- **←** and **→** flip through the deck (or just click a card)
- **↵** shows the answer as soon as it is ready

If you have been reading, the app waits for you rather than yanking the card away
mid-sentence. Press ↵ or click **Read the answer** when you are ready.

You can stop a question at any point — the button you sent it with becomes
**Stop**. Passages already found are kept.

---

## Reading the answer

Every answer has the same four parts:

- **Short answer** — the direct response
- **Recommended action** — what to do
- **Rationale** — why, grounded in the sources
- **Limitations** — what the documents do not cover

At the top is a **grounding badge**:

| Badge | Meaning |
|---|---|
| ✓ **Grounded in document passages** | The answer cites the passages it was built from. |
| ⚠ **Partially grounded** | Passages were found, but the answer text carries no citations. Check it against the sources yourself. |
| ⚠ **No sufficient source** | Nothing in your library answered this. See [below](#when-the-answer-falls-short). |

### Citations

The small numbers — **[1]**, **[2]** — are buttons. Click one and it jumps to
that source, which shows you:

- the passage itself, with the **deciding quote highlighted**
- **the original page**, with the passage boxed on it, for PDF documents
- **why it was selected** — the retrieval model's own reason
- buttons into the full document, and into the Retrieval tab

Every source also appears in the **Sources** panel on the right, grouped by
answer, so you can move between them without scrolling.

> **Always check the sources before acting on an answer.** The app is built to
> make that quick, and every answer carries that reminder at its foot for a
> reason: a model can misread a passage it has correctly retrieved.

---

## Following up

Under the composer are two buttons. You choose which one your next message uses —
the app never guesses.

**Ask about this answer** keeps the passages already found and talks about them.
Use it for *"what about in renal impairment?"*, *"summarise that in one line"*,
*"which of those two is first-line?"*. It does not search again, so it is faster,
and its citations point at the same sources already on your screen.

If you ask it something the passages do not cover, it will say so rather than
guess — that is when you want the other button.

**New search** goes back to the library and retrieves afresh. Use it when the
subject has genuinely changed.

`↵` sends using whichever is selected. `⌘↵` always starts a new search.

### The context meter

Next to those buttons is a small bar. It shows how much of the model's working
memory this conversation is using — the passages, the answer, and the
back-and-forth so far. Click it for the breakdown.

You mostly do not need to think about it. It matters in one case: a long
conversation eventually fills that memory, and when it does the **oldest turns
are dropped** so the passages never are. The app tells you when this happens.
Your citations stay valid because the passages are never what gets dropped.

**Clear the conversation** (in that same panel) gives the model a clean slate
while keeping the answer and its sources. The turns are not deleted: they stay
on screen greyed out, and **Restore this conversation** — in the panel, or on
that answer in the Sources dock — puts them back. Asking something new after
clearing makes it permanent, since restoring then would interleave two
conversations.

### Going back to an earlier answer

Each answer in the **Sources** panel has a **Continue from this answer** button.
It points the composer back at that answer's passages — useful when you asked
three questions and want to return to the second. The Retrieval tab follows it
back too.

---

## When the answer falls short

Two things can happen, and the app is explicit about both.

### "This answer is incomplete"

The model has told you the passages did not cover everything, and named what is
missing — usually a patient detail no document could contain (age, renal
function, allergies).

Type it in and press **Rewrite the answer**. The rewrite:

- may **change the recommendation**, not just add a caveat, if your detail
  warrants it
- tells you **what changed** and why
- adds your context as **a numbered source of its own**, so an answer leaning on
  something you typed shows that it did

Your context is marked clearly as coming from you, not from a document. It can
narrow or rule out options the documents describe; it cannot become clinical
evidence on its own.

Rewrites take about as long as the original answer, and can be stopped.

### "No passage in the library answered this"

Nothing matched. This is a real answer, not a failure — and usually one of three
things:

1. **A filter is on.** Check the filter bar above the composer and the tags in
   the Retrieval tab. This is the most common cause.
2. **The question was too oblique.** Name the condition and the decision
   explicitly.
3. **The library genuinely does not cover it.** The Retrieval tab shows you what
   was considered, so you can tell this apart from the other two.

You can still talk to the app about it — ask *why* it found nothing, or how to
rephrase. It will not answer the clinical question from general knowledge, which
is the entire point.

---

## Looking behind the answer

The **Retrieval** tab is the audit trail.

It offers the same run in **two shapes**, switched with the Graph / Boxes
control:

- **Graph** follows the document's own structure — which is how you see that a
  decision about one section ruled out the whole branch beneath it.
- **Boxes** gives every section area in proportion, so you see how much of the
  library a question actually touched.

**The map** shows every document as a block and every section inside it as a
box. During a question you can watch the boxes change colour as the model
decides about each one:

| Colour | Verdict |
|---|---|
| Green | Retrieved — used to answer |
| Red | Rejected — read and judged irrelevant |
| Grey | Pruned — a whole branch skipped as off-topic |
| Faint | Not read — ranked below the cut for this question |

**The trace** lists those decisions in order, each with the model's reason.
Click any line to open the passage it judged.

**Filters** — the collapsible bar above the composer — narrow what gets searched
*before* the question runs. It stays collapsed with your choices visible, each
removable with its ✕, and a meter showing how much of the library is left. When
that meter turns amber the filter, not the library, is the likely reason an
answer came back thin. **Any answer** searches anything matching one of your
choices; **Every answer** requires all of them — much faster, and it will drop
passages that only matched some.

**Tags** — the pills above the map — highlight documents from a folder and dim
everything else. Click more than one to highlight several. The search box next
to them narrows by section title, and the two work together.

**"Why not?"** — for a section that was *not* selected, the app can explain why
on demand.

---

## Adding your own documents

Point Asepsis at a folder of PDFs: **⚙ Settings → PDF corpus → Change…**, then
ingest. Your existing library is not replaced.

**Sub-folder names become tags.** A layout like:

```
my-documents/
  guidelines/
  internal/
  research/
    preprints/
```

gives you `guidelines`, `internal`, `research` and `preprints` as filter pills,
with no extra work. Documents sitting loose at the top level get no tags — which
is fine, just less filterable.

### Scanned and awkward PDFs

Text-based PDFs work well as they are. **Scanned** documents are read with OCR,
and OCR makes mistakes — a heading read as body text, a caption attached to the
wrong figure, a spurious region over a page number. Those errors flow through to
retrieval: a passage the app never sees correctly is a passage it cannot cite.

For documents that matter and do not come through cleanly, we recommend the
**Asepsis Annotation & Correction Tool**:

**https://github.com/hospital-hygiene-assistant/asepsis-annotation-tool**

![The Asepsis Annotation & Correction Tool: a page of a document with each detected region boxed and labelled — doc_title, paragraph_title, text, figure, caption — a list of regions on the page, a full edit history, and a session summary reporting how many regions were corrected.](assets/annotation-tool.webp)

It shows you every region the layout model found on a page, labelled by what it
thinks that region is, and lets you fix the ones it got wrong — correct a label,
redraw a box, delete a spurious region, add a missed one. Every edit is recorded,
so a corrected document carries its own provenance: who changed what, and how
much of the original the model got right.

We plan to integrate it directly into Asepsis. For now it runs alongside, and
its corrected output can be ingested here.

---

## Settings worth knowing about

Open with the **⚙** button, or **⌘,**.

- **Take the guided tour** — the walk-through, any time.
- **Open the user guide** — this document, rendered in the app.
- **PDF corpus** — which folder your documents come from.
- **Model & prompts** (in the Chatbot tab's header) — read-only, and complete:
  which model answers, over which engine, with the exact text sent to it at
  every step — retrieval, pruning, answering, follow-up and rewrite. Nothing
  about how an answer was produced is hidden from you.
- **Ollama instances** — how many copies of the model run at once. More is
  faster on a machine with memory to spare, and slower on one without. Leave it
  at 1 unless you have reason to change it.
- **Agent context** — how much the answering model can hold at once. Larger means
  more passages are read per question, and a slower answer. The app tells you
  when passages were left unread because this filled up.
- **Flag incomplete answers** — on by default. This is what produces "this answer
  is incomplete" and the rewrite box. Costs a little speed per answer; worth it.
- **Debug cache** — for development. Replays a previous question instead of
  running it again. Off by default; replayed answers are badged **CACHED**.

---

## Keyboard shortcuts

| Key | Does |
|---|---|
| `↵` | Send — or, while the evidence deck is up, show the answer |
| `⇧↵` | New line in the composer |
| `⌘↵` | Force a new search, whichever mode is selected |
| `←` `→` | Flip through the evidence deck, and the guided tour |
| `⌘,` | Settings |
| `Esc` | Close the tour, a dialog, or a full-screen page view |

---

## Common questions

**Does this need the internet?**
No. The model runs on this machine, through a local runtime called Ollama, and
your documents are indexed locally. Pull out the network cable and everything
still works. The only time you need a connection is the first install, to
download the model.

**Is it private?**
Completely. No question, document, or answer leaves your machine. There is no
account, no telemetry, and no external service being called. This is why the app
runs a local model rather than a hosted one — the trade is a slower answer for a
guarantee that nothing is sent anywhere.

**Why does an answer take twenty or thirty seconds?**
Because the model reads the candidate passages one at a time and has to justify
each decision, rather than pattern-matching against an index. That is what makes
the trace and the citations real. The evidence deck exists so the wait is spent
reading rather than watching a spinner.

**Can I add more documents?**
Yes — see [Adding your own documents](#adding-your-own-documents). There is no
document limit; larger libraries make retrieval slower rather than worse.

**Why did it cite a passage that does not quite support the claim?**
It happens, and it is exactly what the citation links are for. A model can
retrieve the right passage and still summarise it loosely. Check the highlighted
quote against the claim; if they disagree, trust the document.

**Can I trust the answer?**
Treat it as a well-read colleague's first pass, not a decision. It is grounded in
your documents and shows its working, which is more than most tools offer — but
it is generated text, it can be wrong, and the sources are one click away
precisely so you can confirm it before acting.
