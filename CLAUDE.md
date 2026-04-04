# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goals

This is a learning project building a RAG application end-to-end. Planned phases:

1. **Local RAG** — PostgreSQL + pgvector as vector store, local website frontend
2. **AWS deployment** — RDS for PostgreSQL, Lambda or EC2 for the app
3. **RAG agent** — evolve the RAG into an agentic application
4. **Graph RAG**

**Current status:** The HTML → text → chunk+embed pipeline is complete. Chunks, embeddings, and metadata are saved in `data/chunks/chunks_collection.jsonl`. Next step: ingest into PostgreSQL pgvector.

**Data source:** 10-K filings are downloaded from SEC EDGAR (https://www.sec.gov/cgi-bin/browse-edgar). Each filing is saved as a single HTML file named `{ticker}-{date}.html` in `data/raw/`.

**Known quirk:** The regex used to extract company name from filings (`\(Exact name of Registrant as specified in its charter\)`) fails on some filings (e.g. `avgo`). Company name extraction is unreliable — use the ticker/doc_id from the filename instead.

**Embedding dimension:** `text-embedding-3-small` produces **1536-dimensional** vectors. Use this when defining the pgvector column.

## Environment Setup

Dependencies are managed with `uv`. Requires Python >=3.11.

```bash
uv sync                        # install dependencies
```

Requires a `.env` file with:
```
OPENAI_KEY=<your openai api key>
```

## Running the Pipeline

```bash
uv run python chunk_embed.py   # chunk and embed all text files in data/text/
uv run jupyter notebook        # open the experiment notebook
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) pipeline over SEC 10-K annual filings. The data flow is:

```
data/raw/*.html  -->  [notebook: Process Raw]  -->  data/text/*.txt
data/text/*.txt  -->  ChunkEmbed (chunk_embed.py)  -->  data/chunks/chunks_collection.jsonl
```

**Stage 1 — HTML to text** (`experiment.ipynb`, "Process Raw" section):
- Parses each 10-K HTML with BeautifulSoup + lxml
- Strips scripts/styles, extracts text, normalizes whitespace
- Truncates everything before "SECURITIES AND EXCHANGE COMMISSION"
- Outputs one `.txt` per filing to `data/text/`

**Stage 2 — Chunk + Embed** (`chunk_embed.py`):
- `ChunkEmbed` class reads `.txt` files from `data/text/`
- Splits each document first by 10-K section headers (Item 1. through Item 16.), then applies LangChain's `RecursiveCharacterTextSplitter` (default 1000 chars, 200 overlap) within each section
- Embeds each chunk with OpenAI `text-embedding-3-small`
- Saves all chunks as JSONL to `data/chunks/chunks_collection.jsonl`

Each chunk in the JSONL has: `chunk_id`, `text`, `embedding` (list of floats), and `metadata` (stock symbol, doc_id, part, section, chunk index, source path, embedding model).

**`experiment.ipynb`** is the prototyping environment — it contains earlier/exploratory versions of both stages. The canonical production code is `chunk_embed.py`.

`langchain-postgres` is a dependency (for pgvector storage), but the PostgreSQL ingestion step is not yet implemented in the committed code.

## Known Issues & Design Notes

**Item 16 is silently dropped** — `chunk_embed.py` loops `range(1, len(SECTION_HEADERS))` so the last header (Item 16) is never used as a section start. Fix before ingesting into postgres.

**Use metadata for filtered retrieval** — chunks carry `stock symbol`, `part`, and `section` metadata, which enables filtered vector search (e.g. restrict to Item 7 MD&A, or a specific ticker). Design the pgvector schema and retrieval layer to leverage these filters.

**Prefer EC2/ECS over Lambda for the app server** — Lambda's cold start latency and 15-minute timeout are problematic for LLM chains, especially once the app becomes agentic. A persistent FastAPI server on EC2 or ECS is simpler to reason about.

**Plan for retrieval evaluation before the agent phase** — without a way to measure whether the right chunks are retrieved, it's hard to diagnose agent failures. A small set of hand-labeled question→expected-chunk pairs is enough to start.

**Graph RAG will likely require re-chunking** — Graph RAG needs entity extraction across documents and typically works better with different chunk sizes and metadata. Keep the current schema flexible enough to accommodate this.
