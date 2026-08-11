"""
Okapi BM25 ranking over candidate leaves.

Self-contained on purpose: the project's only runtime dependency is `ollama`,
and a scoring function this small does not justify pulling in a library.

Its job is to order the leaves that survived pruning so the most promising are
evaluated first. That ordering is what makes the context budget in
`pageindex.evaluate_ranked` meaningful — when the budget runs out, what is
left unevaluated is the tail of the ranking rather than an arbitrary subset.

Determinism matters: the same query over the same corpus must always produce
the same order, including ties, or a cached run and a live run would disagree.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

K1 = 1.5
B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately small. An aggressive list would strip clinically meaningful
# short words; these carry no discriminative weight in any corpus.
STOPWORDS = frozenset("""
a an and are as at be by for from has have how in is it its of on or that the
to was were what when where which who why will with
""".split())


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower())
            if t not in STOPWORDS and len(t) > 1]


@dataclass
class Scored:
    item: object
    score: float
    rank: int


class BM25:
    def __init__(self, documents: list[list[str]]):
        self.docs = documents
        self.n = len(documents)
        self.doc_len = [len(d) for d in documents]
        self.avg_len = (sum(self.doc_len) / self.n) if self.n else 0.0

        self.freqs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in documents:
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self.freqs.append(counts)
            for token in counts:
                df[token] = df.get(token, 0) + 1

        # Lucene-style idf: always positive, so a term appearing in every
        # document contributes ~0 rather than pushing scores negative.
        self.idf = {
            token: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def score(self, query_tokens: list[str], index: int) -> float:
        if not self.n or not self.avg_len:
            return 0.0
        counts = self.freqs[index]
        length = self.doc_len[index]
        total = 0.0
        for token in query_tokens:
            tf = counts.get(token, 0)
            if not tf:
                continue
            denom = tf + K1 * (1 - B + B * length / self.avg_len)
            total += self.idf.get(token, 0.0) * (tf * (K1 + 1)) / denom
        return total


def rank(items: list, query: str, text_of) -> list[Scored]:
    """Rank `items` against `query`, best first.

    `text_of(item) -> str` supplies the text to score. Ties keep the input
    order, so ranking a corpus in document order stays reproducible.
    """
    if not items:
        return []
    docs = [tokenize(text_of(item)) for item in items]
    bm25 = BM25(docs)
    query_tokens = tokenize(query)

    scored = [(i, bm25.score(query_tokens, i)) for i in range(len(items))]
    # -score first, then original index: a stable, total order.
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [Scored(item=items[i], score=score, rank=position)
            for position, (i, score) in enumerate(scored)]
