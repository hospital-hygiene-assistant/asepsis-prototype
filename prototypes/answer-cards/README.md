# Evidence Cards — interlude prototype

Standalone, no dependencies, dummy data. Open `index.html` in any browser, or
drop it into `tauri-app/ui/` unchanged.

`artifact.html` is the same page with the `<html>/<head>/<body>` wrapper
stripped, for publishing only. Edit `index.html`; regenerate the other.

## What it prototypes

Retrieval finishes long before the answer does — measured on this corpus, the
synthesis call spends ~27s reasoning before it emits a single token, then ~7s
writing. That is 27 seconds of a spinner today. This fills it with the evidence
the query already found:

1. **Retrieving** — the live trace, as now.
2. **Cards** — the retrieved passages, one per card, clicked through like a
   deck. The answer composes in the background; the status pill turns
   "Answer ready" and the button changes, but it never yanks you out of a card.
3. **Answer** — typed at ~900 characters/second, well above skim speed, so the
   answer is seen *arriving* rather than pasted.

## Folder colour coding

A document's colour comes from its **outermost** folder, so everything from
`research/` reads as one family whatever the nesting depth; the chip names the
**innermost** folder, so the card still says where it actually came from, and a
nested folder gets a dashed chip border. The colour follows the passage into
the answer: citation chips take the folder colour of the source they point at.

## Integration points

Three functions, marked INTEGRATION at the bottom of the script:

| prototype | real app |
|---|---|
| `renderRetrieving()` | already driven by `/api/status` live events |
| `renderCards(sources)` | `grounding.sources` from `/api/chat`, plus the `folder` field `/api/documents` already returns |
| `showAnswer(text)` | the synthesis result — streamed or not |

The one thing the app cannot do yet: cards need to appear when **retrieval**
finishes, not when the whole `/api/chat` call returns. That needs the answer
and the sources to arrive separately — either the streaming change discussed
earlier, or a smaller split where retrieval returns first.

## Prototype controls

Bottom right: compose time, typing speed, retrieval duration, replay. The
defaults are the measured ones (16s compose, 900 c/s, 3s retrieval).
