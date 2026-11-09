"""Answer questions from the retrieved RBI paragraphs, with citations.

Flow: retrieve top-k chunks -> refuse early if nothing relevant was found ->
ask an LLM to answer ONLY from the numbered context blocks -> keep only
citations that point to real blocks.

Works with any OpenAI-compatible API (OpenAI, Groq, OpenRouter, local Ollama):
    OPENAI_API_KEY, OPENAI_BASE_URL (optional), REGLENS_MODEL (default gpt-4o-mini)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from reglens.retrieve import BM25, Hit

REFUSAL = "The provided RBI directions don't cover this question."
# below this BM25 score the best match is usually just a shared common word
MIN_BM25_SCORE = 4.0
CITE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """You answer questions about RBI (Reserve Bank of India) Master Directions.
Rules:
- Use ONLY the numbered context blocks. Do not use outside knowledge.
- Cite every factual sentence with the block number in square brackets, e.g. [2].
- If the blocks do not contain the answer, reply exactly: NOT_COVERED
- Be concise: 2-6 sentences, keep the regulatory wording where it matters."""


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleLLM:
    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError('answer generation needs: pip install -e ".[llm]"') from e
        self.client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
        self.model = model or os.getenv("REGLENS_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


@dataclass
class Answer:
    question: str
    text: str
    citations: list[dict] = field(default_factory=list)
    refused: bool = False

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.text, "refused": self.refused,
                "citations": self.citations}


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for n, hit in enumerate(hits, 1):
        blocks.append(f"[{n}] ({hit.citation})\n{hit.chunk['text']}")
    return "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def _cited(text: str, hits: list[Hit]) -> tuple[str, list[dict]]:
    """Drop citation markers that don't point to a real block, return the used sources."""
    used: list[int] = []

    def keep(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= len(hits):
            if n not in used:
                used.append(n)
            return m.group(0)
        return ""

    cleaned = CITE.sub(keep, text).strip()
    cites = [{"n": n, "citation": hits[n - 1].citation, "chunk_id": hits[n - 1].chunk["id"]} for n in used]
    return cleaned, cites


class Answerer:
    def __init__(self, bm25: BM25, llm: LLM, searcher=None, k: int = 5,
                 min_score: float = MIN_BM25_SCORE) -> None:
        self.bm25, self.llm, self.k, self.min_score = bm25, llm, k, min_score
        self.searcher = searcher or bm25  # hybrid searcher if available

    def ask(self, question: str, source: str | None = None) -> Answer:
        # refusal gate uses BM25 because its scores are absolute, unlike RRF ranks
        top = self.bm25.search(question, k=1, source=source)
        if not top or top[0].score < self.min_score:
            return Answer(question, REFUSAL, refused=True)

        hits = self.searcher.search(question, k=self.k, source=source)
        raw = self.llm.complete(SYSTEM_PROMPT, build_prompt(question, hits))
        if "NOT_COVERED" in raw:
            return Answer(question, REFUSAL, refused=True)
        text, cites = _cited(raw, hits)
        if not cites:  # an answer with no valid citation is not trusted
            return Answer(question, REFUSAL, refused=True)
        return Answer(question, text, cites)
