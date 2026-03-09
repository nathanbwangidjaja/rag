# DocStack

Chat with your PDFs. FastAPI + Mistral, no LangChain or Pinecone.

**[Demo video](https://www.loom.com/share/09a971a64b8e4e29ac6526a49d8ed1cd)**

## Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# MISTRAL_API_KEY in .env

uvicorn app.main:app --reload
```

`http://localhost:8000` — opens the chat UI.
`http://localhost:8000/graph-ui` — opens the knowledge graph explorer (STRETCH).

## Stack

- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF text extraction
- [NumPy](https://numpy.org/) — vector math + cosine similarity
- [httpx](https://www.python-httpx.org/) — async HTTP client for Mistral API
- [Mistral AI](https://docs.mistral.ai/) — embeddings (`mistral-embed`) + chat (`mistral-small-latest`)
- [uvicorn](https://www.uvicorn.org/) — ASGI server
- [Tailwind CSS](https://tailwindcss.com/) (CDN) — UI styling
- [vis-network](https://visjs.github.io/vis-network/docs/network/) (CDN) — graph visualization
- [Cursor](https://www.cursor.com/) — editor / dev environment
- [Google Stitch](https://stitch.withgoogle.com/) — UI layout prototyping

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
├── graph/               (STRETCH)
│   ├── knowledge_graph.py  in-memory graph w/ JSON persistence
│   ├── extract.py          LLM entity/relationship extraction per chunk
│   ├── query_detect.py     detects if a query should trigger graph search
│   ├── search.py           BFS traversal from matched entities
│   └── routes.py           graph API endpoints
└── static/
    ├── index.html        chat UI
    ├── app.js            upload + chat logic, markdown rendering
    ├── graph.html        knowledge graph explorer UI
    └── graph.js          graph visualization logic (vis-network)
```

## Endpoints

- **POST /ingest** — upload PDFs, parse, chunk, embed, store. Also runs entity/relationship extraction for the knowledge graph. Returns pages/chunks per file.
- **POST /query** — question -> PII check -> intent -> rewrite -> semantic + keyword + graph search -> RRF merge -> LLM answer -> hallucination check. Returns answer + sources + intent + transformed query.
- **GET /graph/stats** — node/edge counts and top entities by mention frequency.
- **GET /graph/data** — full graph in vis-network format for visualization.
- **POST /graph/search** — subgraph around a specific entity with N-hop traversal.

## How it works

1. **Ingest**: PDFs go through per-page text extraction (PyMuPDF), repeated headers/footers get stripped, text gets split into ~512 char chunks at sentence boundaries with 64 char overlap. Each chunk is embedded via Mistral and stored in the numpy vector store. BM25 index rebuilds after every ingest.

2. **Query**: incoming question hits PII regex first (SSNs, credit cards, passports — rejected immediately). Then intent detection classifies it — casual greetings and refused topics (legal/medical advice) get canned responses without touching RAG. Everything else gets a query rewrite for better retrieval, then runs through semantic search (cosine similarity), keyword search (BM25), and optionally graph search (BFS traversal from matched entities). All three result sets merge via Reciprocal Rank Fusion, get deduped and filtered by similarity threshold, then go to the LLM with an intent-specific prompt template. If the query mentions known entities, relationship context from the knowledge graph gets injected into the prompt too.

3. **Post-generation**: the hallucination filter sends the answer + source chunks back to the LLM and asks it to check each sentence for evidence. Unsupported claims get flagged with a warning. Not perfect but catches the obvious stuff.

## Design decisions

- **No LangChain/LlamaIndex** — everything is hand-rolled so the retrieval logic is transparent and tunable.
- **No Pinecone/ChromaDB** — numpy matrix with cosine similarity. Good enough for the scale this targets and keeps dependencies minimal.
- **BM25 from scratch** — TF-IDF with length normalization (k1=1.5, b=0.75), stopword removal, no stemming. Didn't want to pull in rank-bm25 or whoosh.
- **RRF over weighted fusion** — Reciprocal Rank Fusion (k=60), no weight tuning between semantic and keyword scores.
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

## Stretch goal — GraphRAG

I've been working on a GraphRAG system for my Oliver Wyman project and wanted to try a quick prototype of it here too. The idea is to build a knowledge graph from document entities and use graph traversal as a third retrieval signal alongside vector and keyword search. Works best with multi-doc corpora where you need to connect info across files. Single PDF = limited value, but the plumbing is there.

During ingestion, each chunk gets sent to the LLM to pull out structured entities (people, orgs, concepts, locations, dates, metrics) and their relationships. These go into an in-memory graph with JSON persistence. Graph extraction runs in the background after the core pipeline finishes, so it doesn't slow down uploads. At query time, if the question mentions a known entity or uses multi-hop language ("related to", "connected to"), the system does a BFS traversal from those entities, pulls up the chunks they appear in, and feeds that as a third ranked list into RRF. Relationship triples also get injected into the prompt so the LLM has structured context on top of the raw chunks.

There's a separate graph explorer at `/graph-ui` — force-directed visualization with vis-network, nodes colored by type and sized by mention count. You can click through entities, see their connections, and search. It's completely standalone from the chat UI, linked via navigation in the header.

The whole thing is behind a `GRAPH_EXTRACTION_ENABLED` flag. If no graph data exists or no entities match, everything works like before.

## Notes

- All config lives in `app/config.py` — chunk size, overlap, model names, top-k, similarity threshold.
- Vector store persists to `data/vectors.npy` + `data/metadata.json`. Knowledge graph persists to `data/graph_nodes.json` + `data/graph_edges.json`.
- Embeddings batch 16 at a time with rate limit pauses for the Mistral API.
- UI layout prototyped with Google Stitch, then adapted to hook into the actual API.