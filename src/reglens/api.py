"""HTTP API and a minimal web page.

    uvicorn reglens.api:app --port 8000
    open http://localhost:8000          # simple search / ask page
    open http://localhost:8000/docs     # Swagger

/search works without any API key. /ask needs OPENAI_API_KEY (or another
OpenAI-compatible endpoint via OPENAI_BASE_URL).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from reglens.retrieve import BM25, load_chunks

app = FastAPI(title="RegLens", description="Search and Q&A over RBI Master Directions with citations")
STATIC = Path(__file__).parent / "static" / "index.html"


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(5, ge=1, le=20)
    source: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    source: str | None = None


@lru_cache(maxsize=1)
def chunks() -> list[dict]:
    return load_chunks()


@lru_cache(maxsize=1)
def bm25() -> BM25:
    return BM25(chunks())


@lru_cache(maxsize=1)
def answerer():
    from reglens.answer import Answerer, OpenAICompatibleLLM

    return Answerer(bm25(), OpenAICompatibleLLM())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return STATIC.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(chunks()), "llm_configured": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    hits = bm25().search(req.query, k=req.k, source=req.source)
    return {"query": req.query, "results": [
        {"score": round(h.score, 3), "citation": h.citation, "chapter": h.chunk["chapter"],
         "chunk_id": h.chunk["id"], "text": h.chunk["text"]} for h in hits
    ]}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "LLM not configured: set OPENAI_API_KEY (and OPENAI_BASE_URL for other providers)")
    try:
        return answerer().ask(req.question, source=req.source).to_dict()
    except ImportError as e:
        raise HTTPException(503, str(e)) from e
