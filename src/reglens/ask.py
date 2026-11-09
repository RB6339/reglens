"""Ask a question from the terminal.

    export OPENAI_API_KEY=...
    python -m reglens.ask "How often should KYC be updated for high risk customers?"
"""

from __future__ import annotations

import argparse

from reglens.answer import Answerer, OpenAICompatibleLLM
from reglens.retrieve import BM25, load_chunks
from reglens.search import build_searcher


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask RegLens a question about RBI directions")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--source", help="only use one direction, e.g. kyc")
    args = parser.parse_args(argv)

    chunks = load_chunks()
    answerer = Answerer(BM25(chunks), OpenAICompatibleLLM(), searcher=build_searcher(args.mode, chunks))
    ans = answerer.ask(args.question, source=args.source)
    print(ans.text)
    if ans.citations:
        print("\nSources:")
        for c in ans.citations:
            print(f"  [{c['n']}] {c['citation']}")


if __name__ == "__main__":
    main()
