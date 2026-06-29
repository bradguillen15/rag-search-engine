# RAG Search Engine

Guided project from [Boot.dev's Learn RAG course](https://www.boot.dev/courses/learn-retrieval-augmented-generation) — a **Hoopla** movie keyword search engine that builds up to full retrieval-augmented generation.

## What it covers

- **Keyword search** — inverted index over movie metadata
- **Semantic search** — embeddings with sentence-transformers
- **Hybrid search** — RRF fusion of keyword + semantic results
- **Chunking & reranking** — better retrieval for long documents
- **RAG** — Gemini-powered Q&A, summarization, and citations over search results
- **Multimodal search** — image description + search (course extension)
- **Evaluation** — LLM-aspect retrieval quality

## Stack

Python · NLTK · NumPy · sentence-transformers · Google Gemini API · uv

## Setup

```bash
uv sync
cp .env.example .env   # add GEMINI_API_KEY
```

Run CLIs from the `cli/` directory (e.g. `python keyword_search_cli.py`, `python augmented_generation_cli.py rag "your query"`).

## Note

Coursework project — demonstrates RAG pipeline concepts, not a production deployment.
