#!/usr/bin/env python3
"""
inference.py — THE JUDGE'S ENTRY POINT
═══════════════════════════════════════
Bulletproof evaluation script for the BIS x Sigma Squad AI Hackathon.

Usage:
    python inference.py --input queries.json --output results.json

Input format  (queries.json):
    [
        { "id": "q1", "query": "Portland cement for building construction" },
        { "id": "q2", "query": "Steel reinforcement bars for concrete" }
    ]

Output format (results.json):
    [
        {
            "id": "q1",
            "retrieved_standards": ["IS 269: 1989", "IS 455: 1989"],
            "latency_seconds": 1.23
        },
        ...
    ]

Architectural guarantees:
  ✓ Zero-crash: every query is wrapped in try/except → empty list fallback
  ✓ SQLite WAL caching: repeated queries resolve in < 1 ms
  ✓ Strict JSON schema compliance
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("inference")


# ═════════════════════════════════════════════════════════════════════════════
# SQLite WAL Cache
# ═════════════════════════════════════════════════════════════════════════════

class QueryCache:
    """
    SQLite-backed query cache with WAL mode for concurrent read performance.
    Guarantees sub-millisecond lookups for previously-seen queries.
    """

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_cache (
                query_text TEXT PRIMARY KEY,
                standards_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, query: str):
        """Return cached standards list or None."""
        row = self.conn.execute(
            "SELECT standards_json FROM query_cache WHERE query_text = ?",
            (query,),
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def put(self, query: str, standards: list[str]):
        """Insert or replace a query result in the cache."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO query_cache (query_text, standards_json, created_at)
            VALUES (?, ?, ?)
            """,
            (query, json.dumps(standards), time.time()),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


# ═════════════════════════════════════════════════════════════════════════════
# RAG Pipeline (lazy imports to avoid crash if models aren't downloaded yet)
# ═════════════════════════════════════════════════════════════════════════════

def run_rag_pipeline(query: str) -> list[str]:
    """
    Execute the full RAG pipeline: retrieve → rerank → generate.
    Returns a list of BIS standard ID strings.
    """
    from src.retriever import retrieve_standards
    from src.generator import generate_response

    # Step 1: Hybrid retrieval + reranking
    chunks = retrieve_standards(query, top_k=15, final_k=5)

    if not chunks:
        logger.warning("No chunks retrieved for query: %s", query[:60])
        return []

    # Step 2: LLM generation
    standards = generate_response(query, chunks)
    return standards


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BIS Standard Recommendation Engine — Inference Script"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file containing queries",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write output JSON results",
    )
    args = parser.parse_args()

    # ── Load input ───────────────────────────────────────────────────────────
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            queries = json.load(f)
        if not isinstance(queries, list):
            queries = [queries]
        logger.info("Loaded %d queries from %s", len(queries), args.input)
    except Exception as e:
        logger.error("Failed to load input file: %s", e)
        # Even file-read failure must produce valid output
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        sys.exit(1)

    # ── Init cache ───────────────────────────────────────────────────────────
    from src.config import SQLITE_CACHE_PATH

    cache = QueryCache(SQLITE_CACHE_PATH)

    # ── Process queries ──────────────────────────────────────────────────────
    results: list[dict] = []

    for item in queries:
        query_id = item.get("id", "unknown")
        query_text = item.get("query", "")
        start_time = time.time()

        # ── Zero-crash guard ─────────────────────────────────────────────────
        try:
            # Check cache first
            cached = cache.get(query_text)
            if cached is not None:
                logger.info("[CACHE HIT] id=%s", query_id)
                standards = cached
            else:
                logger.info("[CACHE MISS] id=%s — running RAG pipeline", query_id)
                standards = run_rag_pipeline(query_text)
                cache.put(query_text, standards)

        except Exception as e:
            logger.error(
                "Pipeline failed for id=%s: %s", query_id, e, exc_info=True
            )
            standards = []  # ← Zero-crash fallback

        latency = time.time() - start_time

        results.append(
            {
                "id": query_id,
                "retrieved_standards": standards,
                "latency_seconds": round(latency, 4),
            }
        )
        logger.info(
            "id=%s | standards=%d | latency=%.3fs",
            query_id,
            len(standards),
            latency,
        )

    # ── Write output ─────────────────────────────────────────────────────────
    cache.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Results written to %s (%d entries)", args.output, len(results))


if __name__ == "__main__":
    main()
