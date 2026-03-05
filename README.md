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
