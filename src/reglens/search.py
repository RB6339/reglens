"""Search the RBI chunks from the terminal.

    python -m reglens.search "how often should KYC be updated for high risk customers"
    python -m reglens.search -k 3 --source kyc "video based customer identification"
    python -m reglens.search --mode hybrid "how often do I need to redo KYC"   # needs .[dense]
"""

from __future__ import annotations

import argparse
import textwrap

from reglens.retrieve import BM25, load_chunks


def build_searcher(mode: str, chunks: list[dict]):
    if mode == "bm25":
        return BM25(chunks)
    from reglens.dense import EMB_FILE, DenseIndex, HybridSearcher, SentenceTransformerEncoder

    dense = DenseIndex(chunks, SentenceTransformerEncoder(), cache=EMB_FILE)
    return dense if mode == "dense" else HybridSearcher(BM25(chunks), dense)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search over RBI Master Direction chunks")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5, help="number of results")
    parser.add_argument("--source", help="only search one direction, e.g. kyc")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    args = parser.parse_args(argv)

    searcher = build_searcher(args.mode, load_chunks())
    hits = searcher.search(args.query, k=args.k, source=args.source)
    if not hits:
        print("No matching paragraphs.")
        return
    for rank, hit in enumerate(hits, 1):
        print(f"[{rank}] {hit.score:.3f}  {hit.citation}")
        if hit.chunk["chapter"]:
            print(f"    {hit.chunk['chapter']}")
        snippet = " ".join(hit.chunk["text"].split())[:300]
        print(textwrap.indent(textwrap.fill(snippet, 100), "    "))
        print()


if __name__ == "__main__":
    main()
