from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.ingestion.routes import router as ingestion_router
from app.query.routes import router as query_router

app = FastAPI(title="RAG Pipeline")
app.include_router(ingestion_router)
app.include_router(query_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
