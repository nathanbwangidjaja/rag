from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="RAG Pipeline")


@app.get("/health")
def health_check():
    return {"status": "ok"}
