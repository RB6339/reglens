"""Split extracted RBI pages into paragraph-level chunks with citation metadata.

RBI Master Directions are numbered paragraph by paragraph ("18. The Customer Acceptance
Policy shall ...") and grouped into chapters ("Chapter VI - Customer Due Diligence").
Chunking on those boundaries keeps each chunk self-contained and lets the answer cite
"KYC Directions, para 42, page 41" instead of an arbitrary text window.

Usage:
    python -m reglens.chunk          # chunks every data/processed/*.jsonl into data/chunks.jsonl
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from reglens.paths import ROOT

PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_FILE = ROOT / "data" / "chunks.jsonl"

PARA_START = re.compile(r"^(\d{1,3})\.\s+\S")
CHAPTER = re.compile(r"^Chapter\s+([IVXLC]+)\s*[–-]\s*(.+)$", re.IGNORECASE)
SUB_ITEM = re.compile(r"^\((?:\d{1,2}|[a-z]{1,4})\)\s")
# Table-of-contents lines end in dot leaders + a page number: "B. Applicability ........ 4"
TOC_LINE = re.compile(r"\.{5,}\s*\d+\s*$")
# Annexes restart paragraph numbering at 1, so "para 2" alone would be ambiguous
ANNEX = re.compile(r"^Annex\s*[–-]\s*([IVXLC]+)\s*$", re.IGNORECASE)


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    chapter: str
    para: str
    page_start: int
    page_end: int
    text: str
    annex: str = ""  # "I", "II" ... for chunks inside an annex


def _iter_lines(pages: list[dict]):
    for page in pages:
        for line in page["text"].splitlines():
            line = line.strip()
            if line and not TOC_LINE.search(line):
                yield page["page"], line


def _split_long(lines: list[tuple[int, str]], max_chars: int, overlap: int) -> list[list[tuple[int, str]]]:
    """Split an oversized paragraph at sub-item boundaries ((1), (a), (xix) ...), keeping
    `overlap` trailing lines of context from the previous piece."""
    if sum(len(t) for _, t in lines) <= max_chars:
        return [lines]
    pieces, current, size = [], [], 0
    for item in lines:
        starts_sub = bool(SUB_ITEM.match(item[1]))
        if current and size + len(item[1]) > max_chars and (starts_sub or size > 1.5 * max_chars):
            pieces.append(current)
            current = current[-overlap:] if overlap else []
            size = sum(len(t) for _, t in current)
        current.append(item)
        size += len(item[1])
    if current:
        pieces.append(current)
    return pieces


def chunk_document(pages: list[dict], max_chars: int = 1500, overlap: int = 1) -> list[Chunk]:
    if not pages:
        return []
    source, title = pages[0]["source"], pages[0]["title"]
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    chapter, para, annex = "", "preamble", ""
    buf: list[tuple[int, str]] = []

    def flush() -> None:
        for i, piece in enumerate(_split_long(buf, max_chars, overlap)):
            text = "\n".join(t for _, t in piece).strip()
            if len(text) < 40:  # stray headers, lone numbers
                continue
            # the same para label can repeat (e.g. "intro" before each chapter), keep ids unique
            key = f"{source}:annex-{annex.lower()}-{para}" if annex else f"{source}:{para}"
            n = seen.get(key, 0)
            seen[key] = n + 1
            chunks.append(Chunk(
                id=f"{key}:{n}",
                source=source, title=title, chapter=chapter, para=para, annex=annex,
                page_start=piece[0][0], page_end=piece[-1][0], text=text,
            ))

    for page_no, line in _iter_lines(pages):
        m_annex = ANNEX.match(line)
        if m_annex:
            flush()
            buf = []
            annex = m_annex.group(1).upper()
            chapter, para = f"Annex {annex}", "intro"
            continue
        m_ch = CHAPTER.match(line)
        if m_ch and not annex:
            flush()
            buf = []
            chapter = f"Chapter {m_ch.group(1)} - {m_ch.group(2).strip()}"
            # text between a chapter heading and its first numbered paragraph is not the previous para
            para = "intro"
            continue
        m_para = PARA_START.match(line)
        if m_para:
            flush()
            buf = []
            para = m_para.group(1)
        buf.append((page_no, line))
    flush()
    return chunks


def load_pages(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    all_chunks: list[Chunk] = []
    for path in sorted(PROCESSED_DIR.glob("*.jsonl")):
        doc_chunks = chunk_document(load_pages(path))
        print(f"{path.stem}: {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"wrote {len(all_chunks)} chunks -> {CHUNKS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
