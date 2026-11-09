# RegLens

Ask questions about RBI Master Directions and get answers with citations to the exact paragraph.

> Work in progress. Building this step by step: ingestion -> chunking -> retrieval -> answer generation -> evaluation -> API -> deployment.

## Why

RBI regulations (KYC, cards, interest rates, customer conduct) are long PDFs that get updated often. Finding "what does RBI say about X" usually means Ctrl+F across hundreds of pages. RegLens is a RAG (retrieval-augmented generation) app that answers these questions and always points back to the source paragraph.

## Plan

- [x] Download + parse RBI Master Direction PDFs
- [x] Section-aware chunking with metadata (direction, chapter, paragraph number)
- [x] Retrieval (BM25 + embeddings, hybrid with reciprocal rank fusion)
- [ ] Answers with citations, refuse when the documents don't cover the question
- [ ] Evaluation set (retrieval hit rate, answer faithfulness)
- [ ] FastAPI + simple UI
- [ ] Deploy

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage so far

```bash
python -m reglens.ingest            # download + extract the PDFs listed in configs/sources.yaml
python -m reglens.chunk             # paragraph-level chunks -> data/chunks.jsonl
python -m reglens.search "how often should KYC be updated for high risk customers"

# meaning-based and hybrid search (downloads BAAI/bge-small-en-v1.5 on first run)
pip install -e ".[dense]"
python -m reglens.search --mode hybrid "how often do I need to redo KYC"
```

`--mode bm25` is good at exact regulatory terms ("V-CIP", "Officially Valid Document"), `--mode dense` catches
paraphrases, and `--mode hybrid` merges both with reciprocal rank fusion. Which one works best will be measured
in the evaluation step.

Example output:

```
[1] 14.16  Know Your Customer (KYC) Directions, 2025, para 42, p. 41
    Chapter VI - Customer Due Diligence (CDD) Procedure
```
