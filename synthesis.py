"""
Answer synthesis — the ONE place the synthesis prompt lives.

The chat endpoint used to build this prompt inline, while a look-alike module
constant (shown to the user under "prompts" in /api/chat/config) had already
drifted from it: the constant lacked the completeness labels the handler
actually asks for. An evaluation that calls "the synthesis prompt" from a
script has to call the same code the product runs, or it stops describing the
product. So the prompt, the passage formatting, the answer parsing and the
call itself are all here, and both the server and the eval scripts import
them.

Two signals come out of an answer, and they are computed independently:

  grounding   mechanical — did the answer cite any of its sources? Three
              values: no sources at all, cited, uncited. Deterministic, so
              it needs no evaluation.
  judgment    the model's own claim about whether the passages sufficed
              (EVIDENCE_SUFFICIENT) and what is missing (STILL_NEEDED).
              This is the thing worth evaluating.

They never meet: an answer can be badged as grounded while the model says its
evidence was insufficient. `interpret_judgment` keeps the raw label string
next to its interpretation precisely so a hedged phrasing ("Partially") or a
missing label is visible in results rather than silently read as one side.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import ollama

import config as app_config
import tokens as app_tokens


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT_BASE = (
    "You are a clinical knowledge assistant. Answer the question using ONLY the provided source passages. "
    "Every claim must cite its passage inline as [n] (e.g. [1] or [2][3]). Never invent a source number.\n\n"
    "Question: {query}\n\n"
    "Source passages:\n{passages}\n\n"
    "Respond in EXACTLY this format (keep the labels, fill in the text; each section is 1-3 sentences of plain text with inline [n] citations):\n\n"
    "SHORT_ANSWER: <the direct answer to the question>\n"
    "RECOMMENDED_ACTION: <what the clinician should do>\n"
    "RATIONALE: <why, grounded in the cited passages>\n"
    "LIMITATIONS: <what the sources do not cover, or uncertainty>\n\n"
    "If the passages cannot answer the question, say so in SHORT_ANSWER and leave the other sections brief.\n"
    "Do not use markdown headers, bullet points, bolding, or lists. Write ONLY plain text under each of the four labels."
)

# Appended when the completeness check is on. The two labels the UI and the
# evaluation key on live here and nowhere else.
COMPLETENESS_SUFFIX = (
    "\n\nThen add two more labels, on their own lines:\n"
    "EVIDENCE_SUFFICIENT: yes or no — 'no' if the passages leave any"
    " part of the question unanswered, or if the answer depends on"
    " patient details the question did not give.\n"
    "STILL_NEEDED: when 'no', ONE sentence per item, each starting"
    " with '- ', naming what would be needed — a specific document"
    " topic that appears to be missing, or a specific patient detail"
    " you would have to know. Say nothing here when 'yes'."
)


def build_synthesis_prompt(query: str, passages: str,
                           completeness: Optional[bool] = None) -> str:
    """The exact prompt the answering model receives.

    `completeness=None` reads the runtime setting, which is what the app
    does; a script can pin it either way.
    """
    if completeness is None:
        completeness = bool(app_config.runtime().completeness_check)
    prompt = SYNTHESIS_PROMPT_BASE.format(query=query, passages=passages)
    return prompt + (COMPLETENESS_SUFFIX if completeness else "")


def format_passage(source: dict) -> str:
    """One numbered passage block, exactly as the chat handler renders it."""
    label = (source.get("breadcrumb") or source.get("title") or "")
    doc = str(source.get("doc") or "").replace("_", " ")
    return f"[{source['n']}] {doc} › {label}\n{source.get('excerpt') or '(no content)'}"


def format_passages(sources: list[dict]) -> str:
    return "\n\n".join(format_passage(s) for s in sources)


def number_sources(sources: list[dict]) -> list[dict]:
    """Assign 1-based `n` and `id` in list order, returning copies."""
    out = []
    for i, s in enumerate(sources, start=1):
        s = dict(s)
        s["n"] = i
        s["id"] = f"s{i}"
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Parsing the answer
# ---------------------------------------------------------------------------

SECTION_KEYS = ["SHORT_ANSWER", "RECOMMENDED_ACTION", "RATIONALE", "LIMITATIONS",
                "EVIDENCE_SUFFICIENT", "STILL_NEEDED", "WHAT_CHANGED"]


def parse_answer_sections(text: str) -> dict:
    """Parse the labeled sections out of the model's answer. Tolerant of
    markdown bolding and missing sections; anything unmatched is absent."""
    pattern_keys = [k.replace("_", r"[\s_-]?") for k in SECTION_KEYS]
    pattern = re.compile(
        r"^\s*(?:#+\s*)?\**\s*(" + "|".join(pattern_keys) + r")\s*\**\s*:?\s*",
        re.MULTILINE | re.IGNORECASE,
    )
    sections: dict[str, str] = {}
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        raw_key = m.group(1)
        key = raw_key.upper().replace(" ", "_").replace("-", "_")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[key.lower()] = text[m.end():end].strip().strip("*").strip()
    return sections


def cited_numbers(text: str) -> list[int]:
    """Source numbers cited in an answer.

    Both [2][3] and [2, 3] appear in practice — counting only the first form
    made a correctly-cited answer look ungrounded.
    """
    out: list[int] = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text or ""):
        out.extend(int(part) for part in group.split(","))
    return out


def strip_thoughts(text: str) -> str:
    return re.sub(r"<thought>.*?(</thought>|$)", "", text or "", flags=re.S).strip()


# ---------------------------------------------------------------------------
# The two signals
# ---------------------------------------------------------------------------

def grounding_status(answer_text: str, n_sources: int) -> tuple[str, set[int]]:
    """Mechanical grounding: (status, valid cited source numbers).

    insufficient_evidence   nothing was retrieved, so nothing was synthesised
    grounded                at least one in-range inline citation
    partially_grounded      sources exist but the answer cites none of them
    """
    cited = set(cited_numbers(answer_text))
    valid = {n for n in cited if 1 <= n <= n_sources}
    if not n_sources:
        return "insufficient_evidence", set()
    if valid:
        return "grounded", valid
    return "partially_grounded", set()


# What the UI does with the label today (tauri-app/ui/chat.js):
#     /^\s*no\b/i.test(a.evidence_sufficient || '')
# — a prefix match on "no", with everything else, INCLUDING AN ABSENT LABEL,
# reading as sufficient. The evaluation reports that reading alongside a
# stricter one so the two can be compared, rather than replacing it.
_UI_INSUFFICIENT_RE = re.compile(r"^\s*no\b", re.IGNORECASE)
_STRICT_RE = re.compile(r"^\s*\**\s*(yes|no)\b", re.IGNORECASE)


def interpret_judgment(sections: dict) -> dict:
    """The model's sufficiency claim, raw and interpreted.

    label:
      sufficient     the label starts with "yes"
      insufficient   the label starts with "no"
      unparsed       a label was emitted but starts with neither — a hedge
                     ("partially", "mostly yes") or prose
      missing        no EVIDENCE_SUFFICIENT label in the answer at all
    ui_reading:      what the product's prefix match makes of the same text
                     ("insufficient" or "sufficient"); differs from `label`
                     exactly when the loose match would mislead.
    """
    raw = sections.get("evidence_sufficient")
    if raw is None:
        label = "missing"
    else:
        m = _STRICT_RE.match(raw)
        if m is None:
            label = "unparsed"
        else:
            label = "sufficient" if m.group(1).lower() == "yes" else "insufficient"
    ui_reading = "insufficient" if _UI_INSUFFICIENT_RE.match(raw or "") else "sufficient"

    needed_raw = sections.get("still_needed") or ""
    still_needed = [re.sub(r"^\s*[-•*]\s*", "", line).strip()
                    for line in needed_raw.splitlines() if line.strip()]
    return {
        "raw": raw,
        "label": label,
        "ui_reading": ui_reading,
        "still_needed": still_needed,
        "still_needed_raw": needed_raw,
    }


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

class SynthesisCancelled(Exception):
    """Raised mid-stream when the caller's `should_cancel` returns True."""


@dataclass
class SynthesisResult:
    prompt: str
    answer_text: str
    sections: dict
    ms: int
    prompt_eval_count: int = 0
    model: str = ""
    judgment: dict = field(default_factory=dict)


def synthesise(query: str, sources: list[dict], *,
               model: Optional[str] = None,
               client: Optional[ollama.Client] = None,
               url: Optional[str] = None,
               completeness: Optional[bool] = None,
               should_cancel: Optional[Callable[[], bool]] = None) -> SynthesisResult:
    """Run the synthesis call over already-numbered `sources`.

    This is the whole of what the chat endpoint does between "retrieval
    finished" and "answer parsed", so a script that hands it an arbitrary
    passage set is exercising exactly the product's synthesis step and
    nothing else.

    Streamed so a cancel can land mid-answer: breaking out of the iterator
    closes the connection, which is what makes Ollama stop generating.
    """
    cfg = app_config.runtime()
    model = model or cfg.synthesis_model
    if client is None:
        client = ollama.Client(host=url or _default_url())
    passages = format_passages(sources)
    prompt = build_synthesis_prompt(query, passages, completeness)

    started = time.perf_counter()
    chunks: list[str] = []
    prompt_eval = 0
    for part in client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options=app_config.chat_options("agent"),
        keep_alive=app_config.keep_alive(),
        stream=True,
    ):
        if should_cancel is not None and should_cancel():
            raise SynthesisCancelled()
        chunks.append((part.get("message") or {}).get("content") or "")
        prompt_eval = int(part.get("prompt_eval_count") or prompt_eval)
    ms = int((time.perf_counter() - started) * 1000)
    app_tokens.observe(prompt, prompt_eval)

    answer_text = strip_thoughts("".join(chunks))
    sections = parse_answer_sections(answer_text)
    return SynthesisResult(prompt=prompt, answer_text=answer_text,
                           sections=sections, ms=ms,
                           prompt_eval_count=prompt_eval, model=model,
                           judgment=interpret_judgment(sections))


def _default_url() -> str:
    """The first pool URL, without importing pageindex at module load."""
    import pageindex
    return pageindex.OLLAMA_URLS[0]
