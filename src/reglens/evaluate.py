"""Evaluate retrieval (and optionally answers) on a labelled question set.

Each line of eval/kyc_questions.jsonl has a question and the paragraph numbers
that answer it. Out-of-scope questions have no paragraphs and should be refused.

    python -m reglens.evaluate                       # retrieval, BM25
    python -m reglens.evaluate --mode hybrid         # needs .[dense]
    python -m reglens.evaluate --answers             # also run the LLM (needs .[llm] + API key)

Retrieval metrics: hit@1, hit@5 (any expected para in the top k) and MRR.
Answer metrics: refusal accuracy on out-of-scope questions, and how often the
answer cites at least one expected paragraph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reglens.paths import ROOT

EVAL_FILE = ROOT / "eval" / "kyc_questions.jsonl"


def load_eval(path: Path = EVAL_FILE) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieval_metrics(searcher, items: list[dict], k: int = 5) -> dict:
    in_scope = [it for it in items if it["paras"]]
    hit1 = hitk = rr = 0.0
    misses = []
    for it in in_scope:
        ranked = [h.chunk["para"] for h in searcher.search(it["q"], k=k)]
        ranks = [i for i, p in enumerate(ranked, 1) if p in it["paras"]]
        if ranks:
            hit1 += ranks[0] == 1
            hitk += 1
            rr += 1 / ranks[0]
        else:
            misses.append(it["q"])
    n = len(in_scope) or 1
    return {"questions": len(in_scope), "hit@1": round(hit1 / n, 3), f"hit@{k}": round(hitk / n, 3),
            "mrr": round(rr / n, 3), "misses": misses}


def answer_metrics(answerer, items: list[dict]) -> dict:
    oos = [it for it in items if not it["paras"]]
    ins = [it for it in items if it["paras"]]
    refused_ok = sum(answerer.ask(it["q"]).refused for it in oos)
    cited_ok = wrongly_refused = 0
    for it in ins:
        ans = answerer.ask(it["q"])
        if ans.refused:
            wrongly_refused += 1
        elif any(c["chunk_id"].split(":")[1] in it["paras"] for c in ans.citations):
            cited_ok += 1
    return {"out_of_scope_refused": f"{refused_ok}/{len(oos)}",
            "in_scope_cites_expected_para": f"{cited_ok}/{len(ins)}",
            "in_scope_wrongly_refused": f"{wrongly_refused}/{len(ins)}"}


def main(argv: list[str] | None = None) -> None:
    from reglens.retrieve import BM25, load_chunks
    from reglens.search import build_searcher

    parser = argparse.ArgumentParser(description="Evaluate RegLens")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--answers", action="store_true", help="also evaluate LLM answers")
    args = parser.parse_args(argv)

    chunks = load_chunks()
    items = load_eval()
    searcher = build_searcher(args.mode, chunks)
    result = {"mode": args.mode, "retrieval": retrieval_metrics(searcher, items, args.k)}
    if args.answers:
        from reglens.answer import Answerer, OpenAICompatibleLLM

        result["answers"] = answer_metrics(Answerer(BM25(chunks), OpenAICompatibleLLM(), searcher), items)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
