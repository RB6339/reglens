import numpy as np

from reglens.dense import DenseIndex, HybridSearcher, rrf
from reglens.retrieve import BM25, Hit

# tiny fake vocabulary so tests run without downloading a model
VOCAB = ["kyc", "update", "periodic", "video", "identification", "interest", "deposit", "redo"]
SYNONYMS = {"redo": "update", "updation": "update"}


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace(".", " ").split():
                w = SYNONYMS.get(w, w)
                if w in VOCAB:
                    out[i, VOCAB.index(w)] += 1
        return out + 1e-3


def _chunk(cid, text, source="kyc"):
    return {"id": cid, "source": source, "title": "KYC Directions", "chapter": "", "para": cid,
            "page_start": 1, "page_end": 1, "text": text}


CHUNKS = [
    _chunk("38", "Periodic updation of KYC for high risk customers."),
    _chunk("18", "Video based customer identification."),
    _chunk("5", "Interest on deposit accounts.", source="deposit_rates"),
]


def test_dense_finds_paraphrase_bm25_misses():
    dense = DenseIndex(CHUNKS, FakeEncoder(), query_prefix="")
    assert BM25(CHUNKS).search("redo") == []
    assert dense.search("redo", k=1)[0].chunk["id"] == "38"


def test_dense_source_filter():
    dense = DenseIndex(CHUNKS, FakeEncoder(), query_prefix="")
    hits = dense.search("kyc", k=5, source="deposit_rates")
    assert [h.chunk["id"] for h in hits] == ["5"]


def test_embeddings_are_cached_and_rebuilt_when_chunks_change(tmp_path):
    cache = tmp_path / "emb.npy"
    enc = FakeEncoder()
    DenseIndex(CHUNKS, enc, cache=cache)
    DenseIndex(CHUNKS, enc, cache=cache)
    assert enc.calls == 1  # second load came from disk
    DenseIndex(CHUNKS[:2], enc, cache=cache)
    assert enc.calls == 2


def test_rrf_rewards_agreement():
    a, b, c = (_chunk(x, x) for x in "abc")
    fused = rrf([[Hit(9, a), Hit(8, b)], [Hit(0.9, b), Hit(0.8, c)]], k=3)
    assert fused[0].chunk["id"] == "b"
    assert {h.chunk["id"] for h in fused} == {"a", "b", "c"}


def test_hybrid_combines_both():
    hybrid = HybridSearcher(BM25(CHUNKS), DenseIndex(CHUNKS, FakeEncoder(), query_prefix=""))
    ids = [h.chunk["id"] for h in hybrid.search("periodic kyc redo", k=2)]
    assert ids[0] == "38"
