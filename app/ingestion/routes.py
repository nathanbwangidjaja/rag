import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.chunker import chunk_pages
from app.search.embeddings import get_embeddings
from app.search.vector_store import store

router = APIRouter()


@router.post("/ingest")
async def ingest_pdfs(files: list[UploadFile] = File(...)):
    """
    Upload one or more PDFs. Each file gets parsed, chunked, embedded,
    and added to the vector store.
    """
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

    return {
        "ingested": results,
        "total_chunks_in_store": store.count(),
    }
