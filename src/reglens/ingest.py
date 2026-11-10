"""Download RBI Master Direction PDFs and extract clean, page-level text.

Usage:
    python -m reglens.ingest            # download + parse everything in configs/sources.yaml
    python -m reglens.ingest --only kyc
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests
import yaml
from pypdf import PdfReader

from reglens.paths import ROOT

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SOURCES_FILE = ROOT / "configs" / "sources.yaml"

# rbidocs sometimes rejects requests without a browser-like UA
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; reglens/0.1)"}


@dataclass
class Source:
    id: str
    title: str
    url: str


def load_sources(path: Path = SOURCES_FILE) -> list[Source]:
    cfg = yaml.safe_load(path.read_text())
    base = cfg["base_url"].rstrip("/")
    return [Source(s["id"], s["title"], f"{base}/{s['file']}") for s in cfg["sources"]]


def download(src: Source, out_dir: Path = RAW_DIR, retries: int = 3) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{src.id}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(src.url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                raise ValueError(f"{src.url} did not return a PDF")
            dest.write_bytes(resp.content)
            return dest
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                raise
            print(f"  retry {attempt} for {src.id}: {exc}")
            time.sleep(2 * attempt)
    return dest


# --- text cleaning -------------------------------------------------------------

_PAGE_NUM = re.compile(r"^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t]+")


def clean_page(text: str) -> str:
    """Remove page-number lines, join hyphenated line breaks and squash spaces."""
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [ln for ln in text.splitlines() if not _PAGE_NUM.match(ln)]
    lines = [_MULTI_SPACE.sub(" ", ln).strip() for ln in lines]
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def extract_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [clean_page(page.extract_text() or "") for page in reader.pages]


def process(src: Source, out_dir: Path = PROCESSED_DIR) -> Path:
    pdf = download(src)
    pages = extract_pages(pdf)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{src.id}.jsonl"
    with dest.open("w", encoding="utf-8") as f:
        for i, text in enumerate(pages, start=1):
            if text:
                f.write(json.dumps({"source": src.id, "title": src.title, "page": i, "text": text}) + "\n")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="process a single source id")
    args = parser.parse_args()
    for src in load_sources():
        if args.only and src.id != args.only:
            continue
        print(f"[{src.id}] {src.title}")
        out = process(src)
        print(f"  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
