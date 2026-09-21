# Whole-corpus LLM retrieval baseline

The upper bound for the evaluation: a frontier model is given the entire
knowledge base in one context and asked to retrieve the passages relevant to a
question. It is set the *same* task as the system, retrieval rather than
question answering, so that the comparison is retrieval against retrieval and
not retrieval against reading comprehension.

## Files

| File | What it is |
|---|---|
| `build_context.py` | Assembles the corpus prefix from the index. Refuses to write if anything forbidden slips in. |
| `corpus_context.txt` | The finalised cached prefix: 589 passages, 6 documents, ~2.04 M characters. |
| `corpus_manifest.json` | Chunk id to document and node id. **The scoring key. Never sent to the model.** |
| `instructions.txt` | The task instructions, identical for every query, cached with the corpus. |
| `build_gold_map.py` | Resolves each question's gold label to the one leaf that holds it, and verifies it. |
| `gold_map.json` | The result. **Scoring data. Never sent to the model.** |
| `build_prompts.py` | Writes one prompt file per query (instructions + that question). |
| `prompts/` | The per-query prompts, once built. |
| `config.json` | Bedrock settings. `model_id` and pricing are blank until read off the account. |
| `discover_models.py` | Lists what the credential can call, to fill in `model_id`. |
| `run_baseline.py` | Runs the calls. Dry run by default. |
| `results/` | One JSON per call under `raw/`, plus `results.jsonl`. |

## Why the corpus is built from the index, not from the markdown

`knowledge_base/*.md` cannot be pasted in directly. Each chunk is preceded by a
fenced ` ```pin ` block of YAML provenance, and on some chunks that block
carries a `golden: golden-N` line. **That line is the gold label.** A corpus
copied from the markdown would hand the model the answer key for 100 of its
passages.

Building from `index/*.json` avoids this: `pageindex` strips pin blocks and
image lines when it extracts leaf content, so `leaf.content` is already clean.
`build_context.py` asserts it anyway and refuses to write the file if a pin
fence, a `golden:` line, a `bbox:` line or a markdown image line survives.

Passages are labelled with neutral sequential ids (`C0001`...) assigned in
corpus order. They carry no information about which passages are gold.

## Chunking matches the system

The passages are the leaves of the heading tree, in the order the server walks
them: index files sorted by name, then tree order. That is the same unit the
system retrieves, so recall is measured over the same label space.

Each passage shows its breadcrumb, which is the document's own heading path and
is what the system's leaf evaluator sees as "Section path". LLM-written node
summaries are deliberately excluded: they are an artefact of the system under
test, and feeding them to the baseline would make the baseline depend on the
system's own preprocessing.

## Gold labels resolve to exactly one leaf

A gold label lives on a pin, and a pin marks the text directly beneath it, so
every label marks exactly one chunk. `build_gold_map.py` resolves each
question to that chunk and verifies the resolution.

- **48 of the 100** labels sit on a pin attached to a leaf. The gold chunk is
  that leaf.
- **52** sit on a pin attached to an internal section. A section holds no text
  of its own: its prose is the preamble between its heading and its first
  subheading, which `_promote_preambles` lifts into a synthetic
  `<section>-overview` leaf, copying the section's pin onto it. The label
  therefore travels to the overview leaf **and nowhere else**. The section's
  other subsections have their own pins and do not carry the label; the
  builder checks this for all 32 golden sections and fails if any stray is
  found.

Scoring is an **exact match against that one chunk**. It is not an ancestor
rule: crediting any leaf beneath a gold section would inflate recall by up to
95 chunks for the widest gold section in this corpus. A returned passage that
is merely in the right document is recorded as `same_document`, which is
useful for reading a miss but is never counted as a hit.

## Caching, and why the runner does not use Converse

GPT-5.6 Luna supports the Converse API, but **prompt caching for it is
Responses-API-only**. On a 445K-token prefix that is the difference between
about $2.50 and about $20 for 100 queries, so the runner posts to
`/openai/v1/responses` on `bedrock-runtime` and marks the end of the static
content with `prompt_cache_breakpoint: {mode: explicit}`, under
`prompt_cache_options: {mode: explicit, ttl: 30m}`. Explicit mode suppresses
the automatic breakpoint on the latest message, so the cached prefix is
exactly the corpus plus instructions.

Raw HTTP rather than a typed SDK, deliberately: the cache fields are Bedrock
extensions and a client that silently drops an unknown field would leave us
paying the uncached rate while believing otherwise.

The prefix is byte-identical across all calls and carries no timestamp or
per-query id. Every call reports `cached_tokens` and `cache_write_tokens`, and
the runner prints a warning when a call records neither, since that means
caching has stopped engaging.

The TTL is 30 minutes and resets on each hit, so a sequential run stays warm.

## Cost

`estimate_cost.py` prices a run from the config's rate table before anything
is spent. For 100 queries at a 444,659-token prefix on the `us.` profile:

| | with caching | without |
|---|---|---|
| Total | **~$2.48** | ~$19.86 |
| Cache write (once) | $0.24 | n/a |
| Cache reads (99) | $1.94 | n/a |
| Fresh input | ~$0.00 | $19.57 |
| Output | $0.30 | $0.30 |

The `global.` profile is about 9% cheaper. Output volume is the one real
unknown, since reasoning tokens at `effort: high` are billed as output and are
invisible until the first call returns; the range above assumes 500 to 4,000
output tokens per query.

**The corpus sits in Bedrock's long-context pricing tier.** Rates double above
272K input tokens, and the corpus is 445K, so every call is billed at the long
rate. Getting under the threshold would mean cutting roughly 40% of the corpus,
which would stop it being a whole-corpus baseline, so the doubled rate is
accepted rather than engineered around.

## Running it

```bash
python3 -m pip install -r ../requirements-baseline.txt
python3 llm-baseline/build_context.py            # rebuild after any re-index
python3 llm-baseline/build_prompts.py            # the 100 per-query prompts

export AWS_BEARER_TOKEN_BEDROCK=...              # the Bedrock API key
python3 llm-baseline/discover_models.py --filter gpt   # find the model id
# paste it into config.json, then:
python3 llm-baseline/run_baseline.py             # dry run, sends nothing
python3 llm-baseline/run_baseline.py --test A001 # one call
python3 llm-baseline/run_baseline.py --all       # the full set, resumable
```

The credential is read from the environment. It is never written into
`config.json`, into a prompt, or into a result file.

## What each call checks

Beyond whether the gold passage was returned, every call records whether each
returned id actually exists in the corpus, and whether each verbatim quote
really occurs in the passage it was attributed to. A confident id that does not
exist, or a quote that was reworded, is a different failure from a miss and is
counted separately.
