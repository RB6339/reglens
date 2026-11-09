"""Dense (embedding) retrieval and hybrid BM25 + dense search.

BM25 is strong on exact regulatory terms but misses paraphrases: "how often must
I redo KYC" never says "periodic updation". Embeddings cover that gap, and
reciprocal rank fusion (RRF) merges both rankings without having to calibrate
their scores against each other.

Embeddings are cached in data/embeddings.npy and rebuilt when the chunks change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from reglens.retrieve import BM25, Hit

ROOT = Path(__file__).resolve().parents[2]
EMB_FILE = ROOT / "data" / "embeddings.npy"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on queries (not on passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # optional dependency
            raise ImportError('dense search needs: pip install -e ".[dense]"') from e
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


def _fingerprint(chunks: list[dict]) -> str:
    h = hashlib.sha1()
    for c in chunks:
        h.update(c["id"].encode())
        h.update(c["text"].encode())
    return h.hexdigest()[:16]


class DenseIndex:
    def __init__(self, chunks: list[dict], encoder: Encoder, cache: Path | None = None,
                 query_prefix: str = QUERY_PREFIX) -> None:
        self.chunks, self.encoder, self.query_prefix = chunks, encoder, query_prefix
        self.vectors = self._load_or_build(cache)

    def _load_or_build(self, cache: Path | None) -> np.ndarray:
        fp = _fingerprint(self.chunks)
        meta = cache.with_suffix(".json") if cache else None
        if cache and cache.exists() and meta.exists() and json.loads(meta.read_text()).get("fingerprint") == fp:
            return np.load(cache)
        passages = [f"{c.get('chapter', '')}\n{c['text']}" for c in self.chunks]
        vecs = _normalize(self.encoder.encode(passages))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, vecs)
            meta.write_text(json.dumps({"fingerprint": fp, "n": len(self.chunks)}))
        return vecs

    def search(self, query: str, k: int = 5, source: str | None = None) -> list[Hit]:
        q = _normalize(self.encoder.encode([self.query_prefix + query]))[0]
        scores = self.vectors @ q
        order = np.argsort(-scores)
        hits = []
        for i in order:
            c = self.chunks[i]
            if source and c["source"] != source:
                continue
            hits.append(Hit(float(scores[i]), c))
            if len(hits) == k:
                break
        return hits


def rrf(rankings: list[list[Hit]], k: int = 5, c: int = 60) -> list[Hit]:
    """Reciprocal rank fusion: score = sum(1 / (c + rank)) over every ranking a chunk appears in."""
    fused: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            cid = hit.chunk["id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (c + rank)
            by_id[cid] = hit.chunk
    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [Hit(score, by_id[cid]) for cid, score in top]


class HybridSearcher:
    def __init__(self, bm25: BM25, dense: DenseIndex, pool: int = 20) -> None:
        self.bm25, self.dense, self.pool = bm25, dense, pool

    def search(self, query: str, k: int = 5, source: str | None = None) -> list[Hit]:
        return rrf([self.bm25.search(query, self.pool, source), self.dense.search(query, self.pool, source)], k=k)
