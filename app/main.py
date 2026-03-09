import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from app.ingestion.routes import router as ingestion_router
from app.query.routes import router as query_router
from app.graph.routes import router as graph_router

app = FastAPI(title="DocStack")
app.include_router(ingestion_router)
app.include_router(query_router)
app.include_router(graph_router)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/graph-ui")
def serve_graph_ui():
    return FileResponse(os.path.join(STATIC_DIR, "graph.html"))


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health_check():
    return {"status": "ok"}
