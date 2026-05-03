"""
src/retriever.py - Hybrid retrieval pipeline.

Steps:
  1. Dense vector search via ChromaDB (cosine similarity)
  2. Sparse keyword search via BM25 over the full corpus
  3. Reciprocal Rank Fusion to merge both result sets
  4. Cross-Encoder reranking (ms-marco-MiniLM-L-6-v2) to produce final top-k

Performance notes:
  - Cross-encoder input is truncated to 512 chars (model context limit)
  - Rerank candidate cap of 10 keeps CPU inference under 1s
  - Models are lazy-loaded singletons; call warm_up() for eager initialisation
"""

import logging
import time
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import (
    BM25_TOP_K,
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    RERANK_FINAL_K,
    RERANKER_MODEL,
    VECTOR_TOP_K,
)

logger = logging.getLogger(__name__)

_chroma_collection = None
_bm25_index: Optional[BM25Okapi] = None
_bm25_corpus: list[dict] = []
_cross_encoder: Optional[CrossEncoder] = None

_RERANK_TEXT_LIMIT = 512
_RERANK_CANDIDATE_CAP = 10


def _get_collection():
    """Return the ChromaDB collection, initialising it on first call."""
    global _chroma_collection
    if _chroma_collection is None:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _chroma_collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' loaded (%d docs)",
            CHROMA_COLLECTION,
            _chroma_collection.count(),
        )
    return _chroma_collection


def _build_bm25_index():
    """Build a BM25 index from the ChromaDB corpus (runs once per session)."""
    global _bm25_index, _bm25_corpus
    col = _get_collection()
    results = col.get(include=["documents", "metadatas"])
    _bm25_corpus = [
        {"id": doc_id, "text": doc, "metadata": meta}
        for doc_id, doc, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        )
    ]
    tokenized = [doc["text"].lower().split() for doc in _bm25_corpus]
    _bm25_index = BM25Okapi(tokenized)
    logger.info("BM25 index built over %d documents", len(_bm25_corpus))


def _get_cross_encoder() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker model."""
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder: %s", RERANKER_MODEL)
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


def warm_up():
    """Eagerly initialise all models so the first query does not pay load cost."""
    t0 = time.time()
    _get_collection()
    _build_bm25_index()
    _get_cross_encoder()
    logger.info("Retriever warm-up complete in %.1fs", time.time() - t0)


def _vector_search(query: str, top_k: int = VECTOR_TOP_K) -> list[dict]:
    """Dense retrieval via ChromaDB cosine similarity."""
    col = _get_collection()
    results = col.query(query_texts=[query], n_results=min(top_k, col.count()))
    hits: list[dict] = []
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "score": 1.0 - dist,  # cosine distance to similarity
                "source": "vector",
            }
        )
    return hits


def _bm25_search(query: str, top_k: int = BM25_TOP_K) -> list[dict]:
    """Sparse keyword retrieval via BM25."""
    if _bm25_index is None:
        _build_bm25_index()
    tokens = query.lower().split()
    scores = _bm25_index.get_scores(tokens)
    scored = sorted(zip(_bm25_corpus, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "id": doc["id"],
            "text": doc["text"],
            "metadata": doc["metadata"],
            "score": float(s),
            "source": "bm25",
        }
        for doc, s in scored
        if s > 0
    ]


def _reciprocal_rank_fusion(*result_lists: list[dict], k: int = 60) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}
    for result_list in result_lists:
        for rank, hit in enumerate(result_list):
            doc_id = hit["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = hit
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{**doc_map[doc_id], "rrf_score": score} for doc_id, score in fused]


def _rerank(query: str, candidates: list[dict], final_k: int) -> list[dict]:
    """
    Rerank candidates using the cross-encoder model.
    Input text is truncated to _RERANK_TEXT_LIMIT chars to keep CPU inference fast.
    """
    if not candidates:
        return []
    candidates = candidates[:_RERANK_CANDIDATE_CAP]
    encoder = _get_cross_encoder()
    pairs = [[query, c["text"][:_RERANK_TEXT_LIMIT]] for c in candidates]
    scores = encoder.predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:final_k]


def retrieve_standards(
    query: str,
    top_k: int = VECTOR_TOP_K,
    final_k: int = RERANK_FINAL_K,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline: vector + BM25 retrieval, RRF fusion,
    cross-encoder reranking, returning the top final_k results.

    Each result is a dict with keys: id, text, metadata, rerank_score.
    """
    t0 = time.time()
    logger.info("Retrieving standards for: '%s'", query[:80])

    vec_hits = _vector_search(query, top_k=top_k)
    bm25_hits = _bm25_search(query, top_k=top_k)
    logger.info("Vector hits: %d | BM25 hits: %d", len(vec_hits), len(bm25_hits))

    fused = _reciprocal_rank_fusion(vec_hits, bm25_hits)
    logger.info("Fused candidates: %d", len(fused))

    top_results = _rerank(query, fused, final_k)
    logger.info(
        "Reranked results: %d (retrieval took %.2fs)",
        len(top_results),
        time.time() - t0,
    )
    return top_results
