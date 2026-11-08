"""Search the RBI chunks from the terminal.

    python -m reglens.search "how often should KYC be updated for high risk customers"
    python -m reglens.search -k 3 --source kyc "video based customer identification"
"""

from __future__ import annotations

import argparse
import textwrap

from reglens.retrieve import BM25, load_chunks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BM25 search over RBI Master Direction chunks")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5, help="number of results")
    parser.add_argument("--source", help="only search one direction, e.g. kyc")
    args = parser.parse_args(argv)

    index = BM25(load_chunks())
    hits = index.search(args.query, k=args.k, source=args.source)
    if not hits:
        print("No matching paragraphs.")
        return
    for rank, hit in enumerate(hits, 1):
        print(f"[{rank}] {hit.score:.2f}  {hit.citation}")
        if hit.chunk["chapter"]:
            print(f"    {hit.chunk['chapter']}")
        snippet = " ".join(hit.chunk["text"].split())[:300]
        print(textwrap.indent(textwrap.fill(snippet, 100), "    "))
        print()


if __name__ == "__main__":
    main()
