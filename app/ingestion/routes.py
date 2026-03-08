import os
import shutil
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.chunker import chunk_pages
from app.search.embeddings import get_embeddings
from app.search.vector_store import store
from app.search.bm25 import bm25_index
from app.graph.knowledge_graph import graph
from app.graph.extract import extract_entities_and_relationships
from app.config import GRAPH_EXTRACTION_ENABLED

router = APIRouter()


@router.post("/ingest")
async def ingest_pdfs(files: list[UploadFile] = File(...)):
    """Upload PDFs. Parse, chunk, embed, store. Runs graph extraction in background if enabled."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    results = []

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            results.append({"file": f.filename, "status": "skipped", "reason": "not a PDF"})
            continue

        # save upload to disk so pymupdf can read it
        path = os.path.join(UPLOAD_DIR, f.filename)
        with open(path, "wb") as out:
            shutil.copyfileobj(f.file, out)

        try:
            pages = parse_pdf(path)
            if not pages:
                results.append({"file": f.filename, "status": "skipped", "reason": "no text found"})
                continue

            chunks = chunk_pages(pages, source=f.filename, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            texts = [c["text"] for c in chunks]

            embeddings = await get_embeddings(texts)
            store.add(chunks, embeddings)

            results.append({
                "file": f.filename,
                "status": "ok",
                "pages": len(pages),
                "chunks": len(chunks),
            })
        except Exception as e:
            results.append({"file": f.filename, "status": "error", "reason": str(e)})

    # rebuild keyword index with all chunks so far
    bm25_index.build(store.chunks)

    # graph extraction — runs in background so the upload response returns immediately
    if GRAPH_EXTRACTION_ENABLED:
        all_new_chunks = []
        for r in results:
            if r["status"] == "ok":
                count = r["chunks"]
                all_new_chunks.extend(store.chunks[-count:])

        if all_new_chunks:
            asyncio.create_task(_extract_graph(all_new_chunks))

    return {
        "ingested": results,
        "total_chunks_in_store": store.count(),
    }


async def _extract_graph(chunks: list[dict]):
    """Build knowledge graph from chunks in the background."""
    for chunk in chunks:
        try:
            extraction = await extract_entities_and_relationships(chunk)
            graph.add_extraction(
                chunk_id=chunk["id"],
                entities=extraction.get("entities", []),
                relationships=extraction.get("relationships", []),
            )
        except Exception:
            pass  # one chunk failing won't stop the rest
