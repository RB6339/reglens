"""BM25 keyword retrieval over the paragraph chunks.

A small, dependency-free Okapi BM25. Regulatory text is full of exact terms
("Periodic Updation", "V-CIP", "Officially Valid Document") where keyword
matching is hard to beat, so this is the first retriever. Embeddings come next
and get combined with it.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from reglens.chunk import CHUNKS_FILE

# keep numbers and hyphenated terms like "v-cip" or "re-kyc" as single tokens
TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

_STOP = """a an and are as at be been by can do does for from has have how i if in into is it its
of on or shall should such than that the their them then there these this those to under
was what when where which who will with within would"""
STOPWORDS = frozenset(_STOP.split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Hit:
    score: float
    chunk: dict

    @property
    def citation(self) -> str:
        c = self.chunk
        pages = str(c["page_start"]) if c["page_start"] == c["page_end"] else f"{c['page_start']}-{c['page_end']}"
        para = f"para {c['para']}" if c["para"] != "preamble" else "preamble"
        return f"{c['title']}, {para}, p. {pages}"


class BM25:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        # index the chapter heading too, it carries useful terms ("Customer Due Diligence")
        self.docs = [Counter(tokenize(f"{c.get('chapter', '')}\n{c['text']}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        doc, dl = self.docs[i], self.lengths[i]
        s = 0.0
        for t in query_tokens:
            tf = doc.get(t)
            if not tf:
                continue
            s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, k: int = 5, source: str | None = None) -> list[Hit]:
        q = tokenize(query)
        if not q:
            return []
        hits = []
        for i, c in enumerate(self.chunks):
            if source and c["source"] != source:
                continue
            s = self.score(q, i)
            if s > 0:
                hits.append(Hit(s, c))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


def load_chunks(path: Path = CHUNKS_FILE) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run `python -m reglens.chunk` first")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
