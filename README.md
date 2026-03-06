# RAG Pipeline

A from-scratch retrieval-augmented generation system for chatting with PDF documents. Built with FastAPI and Mistral AI, without relying on external RAG frameworks or vector databases.

## How to Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# add your MISTRAL_API_KEY to .env

uvicorn app.main:app --reload
```

Server runs at `http://localhost:8000`.

## Tech Stack

| Library | Why |
|---------|-----|
| [FastAPI](https://fastapi.tiangolo.com/) | API framework |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | PDF text extraction |
| [NumPy](https://numpy.org/) | Vector math / cosine similarity |
| [httpx](https://www.python-httpx.org/) | HTTP client for Mistral API calls |
| [Mistral AI](https://docs.mistral.ai/) | Embeddings (`mistral-embed`) and generation (`mistral-small-latest`) |
| [uvicorn](https://www.uvicorn.org/) | ASGI server |

## Project Structure

```
app/
├── main.py                  # FastAPI app entry point
├── config.py                # env vars, model names, chunking params
├── ingestion/
│   ├── pdf_parser.py        # extract text from PDFs page-by-page (PyMuPDF)
│   ├── chunker.py           # split text into overlapping chunks for embedding
│   └── routes.py            # POST /ingest endpoint
├── query/
│   ├── intent.py            # classifies what the user is actually asking for
│   └── transform.py         # rewrites vague queries into better search terms
└── search/
    ├── embeddings.py        # calls Mistral embed API, handles batching
    ├── vector_store.py      # in-memory numpy vector store, persists to disk
    └── bm25.py              # keyword search, built from scratch
```

### What's wired up so far

**PDF parsing** — `pdf_parser.py` pulls text out per page and strips repeated headers/footers (lines that show up on 60%+ of pages are almost always noise). No OCR support yet, so scanned PDFs won't work.

**Chunking** — `chunker.py` splits on sentence boundaries instead of fixed character counts. Chunks target ~512 chars with 64-char overlap so we don't lose context at the edges. Each chunk tracks which file and page it came from.

**Embeddings** — `embeddings.py` wraps the Mistral embed endpoint. Batches up to 16 texts per call with a short pause between batches to stay under rate limits.

**Vector store** — `vector_store.py` is a bare-bones numpy store. Vectors live in memory as a single matrix; search is brute-force cosine similarity. Good enough for thousands of chunks. Saves to `data/vectors.npy` + `data/metadata.json` so nothing is lost on restart.

**Ingestion endpoint** — `POST /ingest` accepts one or more PDF uploads. Each file gets parsed, chunked, embedded via Mistral, and stored. Non-PDFs are skipped, and you get back per-file stats (pages found, chunks created).

**Intent detection** — before we hit the knowledge base, `intent.py` asks Mistral to figure out what the user actually wants. Greetings and small talk skip retrieval entirely. Queries asking for legal/medical advice or containing PII get refused. Everything else gets routed to the right prompt template (factual, list, summary).

**Query rewriting** — `transform.py` takes vague questions and rewrites them so they embed better. Something like "what does it say about that?" on its own is useless for search, so we expand it into something more specific before embedding.

**BM25 keyword search** - `bm25.py` is a from-scratch implementation. Tokenizes, removes stopwords, scores with TF-IDF and length normalization. No stemming for now (so "cats" won't match "cat"), but it catches things embeddings tend to miss - exact names, acronyms, numbers. The index rebuilds after every ingest call.

### Still to do

- Result merging (reciprocal rank fusion) and re-ranking
- LLM generation with prompt templates
- Guardrails: citation thresholds, hallucination checks, PII refusal
- Chat UI
- Stretch Goal: GraphRAG!