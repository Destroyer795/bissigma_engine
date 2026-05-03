"""
app_api.py - FastAPI microservice backend for the BIS Recommendation Engine.

Exposes REST endpoints for the hybrid RAG recommendation pipeline.
"""

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="BIS Recommendation Engine",
    description="AI-powered Bureau of Indian Standards recommendation API using Hybrid RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Product/material description")
    top_k: int = Field(
        default=5, ge=1, le=10, description="Number of standards to return"
    )


class RecommendResponse(BaseModel):
    query: str
    retrieved_standards: list[str]
    latency_seconds: float
    num_context_chunks: int


class IngestRequest(BaseModel):
    pdf_path: Optional[str] = Field(default=None, description="Path to dataset PDF")


class IngestResponse(BaseModel):
    status: str
    chunks_ingested: int
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for Docker and load balancers."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendation"])
async def recommend_standards(req: RecommendRequest):
    """
    Primary recommendation endpoint.
    Runs the hybrid RAG pipeline: vector + BM25 retrieval, cross-encoder reranking,
    and dual-agent LLM generation.
    """
    start = time.time()

    try:
        from src.retriever import retrieve_standards
        from src.generator import generate_response

        chunks = retrieve_standards(req.query, final_k=req.top_k)
        standards = generate_response(req.query, chunks)
        latency = time.time() - start

        return RecommendResponse(
            query=req.query,
            retrieved_standards=standards,
            latency_seconds=round(latency, 4),
            num_context_chunks=len(chunks),
        )

    except Exception as e:
        logger.error("Recommendation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/ingest", response_model=IngestResponse, tags=["Data"])
async def ingest_data(req: IngestRequest):
    """
    Trigger data ingestion: parse PDF, regex chunk at IS boundaries,
    embed with sentence-transformers, and persist to ChromaDB.
    """
    try:
        from src.ingest import run_ingestion

        count = run_ingestion(req.pdf_path)
        return IngestResponse(
            status="success",
            chunks_ingested=count,
            message=f"Successfully ingested {count} standard chunks.",
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Ingestion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API documentation link."""
    return {
        "service": "BIS Recommendation Engine",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
