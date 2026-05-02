"""
src/config.py
─────────────
Centralised configuration loaded from environment variables.
All sensitive keys are read from a `.env` file via python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PDF_PATH = os.getenv("DATASET_PDF_PATH", str(DATA_DIR / "dataset.pdf"))

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chromadb"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "bis_standards")

# ── SQLite Cache (WAL mode) ──────────────────────────────────────────────────
SQLITE_CACHE_PATH = os.getenv("SQLITE_CACHE_PATH", str(DATA_DIR / "cache.db"))

# ── API Keys ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

# ── Model IDs ────────────────────────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Retrieval Hyper-parameters ───────────────────────────────────────────────
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))
RERANK_FINAL_K = int(os.getenv("RERANK_FINAL_K", "5"))
