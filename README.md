# DocStack

Chat with your PDFs. FastAPI + Mistral, no LangChain or Pinecone.

## Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# MISTRAL_API_KEY in .env

uvicorn app.main:app --reload
```

`http://localhost:8000` — opens the chat UI.

## Stack

- FastAPI, PyMuPDF, NumPy, httpx, Mistral (embed + chat), uvicorn
- Frontend: vanilla JS, Tailwind (CDN), no framework

## Layout

```
app/
├── main.py              entry point, mounts routes + static
├── config.py            all tunables (chunk size, models, thresholds)
├── ingestion/
│   ├── pdf_parser.py    per-page extraction, header/footer removal
│   ├── chunker.py       sentence-aware splitting w/ overlap
│   └── routes.py        POST /ingest
├── query/
│   ├── intent.py        LLM classifies intent (knowledge/list/summary/casual/refuse)
│   ├── transform.py     LLM rewrites question for better retrieval
│   └── routes.py        POST /query — orchestrates the whole pipeline
├── generation/
│   ├── llm.py           thin wrapper around Mistral chat
│   ├── prompts.py       intent-specific templates, canned responses
│   └── filters.py       PII regex, post-hoc hallucination check
├── search/
│   ├── embeddings.py    Mistral embed API w/ batching
│   ├── vector_store.py  numpy cosine similarity store, persists to data/
│   ├── bm25.py          BM25 from scratch, no external search libs
│   └── ranker.py        RRF fusion, dedup, threshold filtering
└── static/
    ├── index.html        chat UI
    └── app.js            upload + chat logic, markdown rendering
```

## Endpoints

- **POST /ingest** — upload PDFs, parse, chunk, embed, store. Returns pages/chunks per file.
- **POST /query** — question → PII check → intent → rewrite → semantic + keyword search → RRF merge → LLM answer → hallucination check. Returns answer + sources + intent + transformed query.

## How it works

1. **Ingest**: PDFs go through per-page text extraction (PyMuPDF), repeated headers/footers get stripped, text gets split into ~512 char chunks at sentence boundaries with 64 char overlap. Each chunk is embedded via Mistral and stored in the numpy vector store. BM25 index rebuilds after every ingest.

2. **Query**: incoming question hits PII regex first (SSNs, credit cards, passports — rejected immediately). Then intent detection classifies it — casual greetings and refused topics (legal/medical advice) get canned responses without touching RAG. Everything else gets a query rewrite for better retrieval, then runs through both semantic search (cosine similarity) and keyword search (BM25). Results merge via Reciprocal Rank Fusion, get deduped and filtered by similarity threshold, then go to the LLM with an intent-specific prompt template.

3. **Post-generation**: the hallucination filter sends the answer + source chunks back to the LLM and asks it to check each sentence for evidence. Unsupported claims get flagged with a warning. Not perfect but catches the obvious stuff.

## Design decisions

- **No LangChain/LlamaIndex** — everything is hand-rolled so the retrieval logic is transparent and tunable.
- **No Pinecone/ChromaDB** — numpy matrix with cosine similarity. Good enough for the scale this targets and keeps dependencies minimal.
- **BM25 from scratch** — TF-IDF with length normalization (k1=1.5, b=0.75), stopword removal, no stemming. Didn't want to pull in rank-bm25 or whoosh.
- **RRF over weighted fusion** — Reciprocal Rank Fusion (k=60) doesn't need weight tuning between semantic and keyword scores, which is nice when you don't have a labeled eval set.
- **Sentence-aware chunking** — splitting on sentence boundaries instead of fixed windows keeps chunks coherent for embedding quality.
- **Intent routing** — different prompt templates for factual Q&A vs lists vs summaries. Casual and refused intents skip retrieval entirely so we don't waste API calls.

## Guardrails

- **PII**: regex catches SSNs, credit cards, passport numbers before anything else runs.
- **Hallucination filter**: LLM-based evidence check post-generation. Flags sentences that can't be backed by source chunks.
- **Refuse**: legal advice, medical questions, PII requests get a polite refusal.
- **Insufficient evidence**: if no chunks pass the similarity threshold, says so instead of making things up.

## Limitations

- No OCR — scanned PDFs won't work.
- Vector store is in-memory + flat file. Fine for hundreds of docs, wouldn't scale to millions.
- BM25 index rebuilds fully on each ingest (no incremental update).
- Hallucination filter is best-effort. It asks the same LLM to check itself, so it has blind spots.

## Stretch goal

- **GraphRAG** — build a knowledge graph from extracted entities/relationships and use graph traversal alongside vector search for multi-hop reasoning queries.

## Notes

- All config lives in `app/config.py` — chunk size, overlap, model names, top-k, similarity threshold.
- Vector store persists to `data/vectors.npy` + `data/metadata.json`.
- Embeddings batch 16 at a time with rate limit pauses for the Mistral API.
- UI layout prototyped with Google Stitch, then adapted to hook into the actual API.