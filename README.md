# RAG Pipeline

Chat with your PDFs. FastAPI + Mistral, no LangChain or Pinecone.

## Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# MISTRAL_API_KEY in .env

uvicorn app.main:app --reload
```

`http://localhost:8000`

## Stack

- FastAPI, PyMuPDF, NumPy, httpx, Mistral (embed + chat), uvicorn

## Layout

```
app/
├── main.py, config.py
├── ingestion/     pdf parse, chunk, POST /ingest
├── query/         intent, rewrite, POST /query
├── generation/    llm + prompts
└── search/        embeddings, vector store, BM25, ranker
```

## Endpoints

- **POST /ingest** — upload PDFs, parse, chunk, embed, store. Returns pages/chunks per file.
- **POST /query** — question → intent → rewrite → semantic + keyword search → RRF merge → LLM answer. Returns answer + sources + intent + transformed query.

## Notes

- PDFs: per-page extraction, strips repeated headers/footers. Scanned PDFs (OCR) not supported.
- Chunks: sentence boundaries, ~512 chars, 64 overlap. Metadata: source, page.
- Vector store: numpy matrix, cosine similarity, persists to data/
- BM25: custom impl, no stemming. Rebuilds on each ingest.
- Ranker: RRF fuses semantic + keyword lists, dedupes overlap, filters by similarity threshold.
- Guardrails: PII regex catches SSNs/credit cards before anything runs. Hallucination filter asks the LLM to check its own answer against the source chunks post-generation — sentences it can't back up get flagged. Not bulletproof but catches the obvious stuff.
- Intent routing picks different prompt templates (factual, list, summary) and skips RAG entirely for greetings or refused queries (legal/medical/PII).