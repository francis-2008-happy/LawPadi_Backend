import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router

app = FastAPI(title="LawPadi", description="Nigerian Legal RAG System", version="1.0.0")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "LawPadi",
        "docs": "/docs",
        "query": "/query",
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/favicon.ico")
def favicon():
    # Avoid noisy 404s in logs; no icon provided.
    return {"status": "ok"}

_cors_origins_raw = os.getenv("CORS_ORIGINS", "").strip()
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()] if _cors_origins_raw else []

if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query_router, prefix="/query", tags=["Query"])
