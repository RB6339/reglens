from fastapi.testclient import TestClient

from reglens import api
from reglens.evaluate import retrieval_metrics
from reglens.retrieve import BM25

CHUNKS = [
    {"id": "kyc:42:0", "source": "kyc", "title": "KYC Directions", "chapter": "Chapter VI - CDD",
     "para": "42", "page_start": 41, "page_end": 41,
     "text": "Periodic updation of KYC at least once every two years for high risk customers."},
    {"id": "kyc:45:0", "source": "kyc", "title": "KYC Directions", "chapter": "Chapter VI - CDD",
     "para": "45", "page_start": 46, "page_end": 46,
     "text": "Accounts of politically exposed persons need senior management approval."},
]


def _client(monkeypatch):
    api.chunks.cache_clear()
    api.bm25.cache_clear()
    monkeypatch.setattr(api, "load_chunks", lambda: CHUNKS)
    return TestClient(api.app)


def test_health_and_index(monkeypatch):
    client = _client(monkeypatch)
    assert client.get("/health").json()["chunks"] == 2
    assert "RegLens" in client.get("/").text


def test_search_endpoint(monkeypatch):
    res = _client(monkeypatch).post("/search", json={"query": "politically exposed persons", "k": 1}).json()
    assert res["results"][0]["citation"] == "KYC Directions, para 45, p. 46"


def test_search_validates_input(monkeypatch):
    assert _client(monkeypatch).post("/search", json={"query": "x"}).status_code == 422


def test_ask_without_key_returns_503(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res = _client(monkeypatch).post("/ask", json={"question": "periodic updation"})
    assert res.status_code == 503 and "OPENAI_API_KEY" in res.json()["detail"]


def test_retrieval_metrics():
    items = [
        {"q": "periodic updation high risk", "paras": ["42"]},
        {"q": "politically exposed", "paras": ["45"]},
        {"q": "gold loan ratio", "paras": [], "out_of_scope": True},
    ]
    m = retrieval_metrics(BM25(CHUNKS), items, k=5)
    assert m["questions"] == 2 and m["hit@1"] == 1.0 and m["mrr"] == 1.0
