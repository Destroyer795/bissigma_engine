"""
src/ingest.py
─────────────
Data ingestion pipeline:
  1. Parse the BIS dataset PDF via LlamaParse (markdown mode).
  2. Regex-chunk the markdown at every "SUMMARY OF IS" boundary.
  3. Extract standard IDs (e.g. "IS 269: 1989") into metadata.
  4. Embed chunks and persist into a local ChromaDB collection.
"""

import re
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from llama_parse import LlamaParse

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    DATASET_PDF_PATH,
    EMBEDDING_MODEL,
    LLAMA_CLOUD_API_KEY,
)

logger = logging.getLogger(__name__)

# ── Regex patterns ───────────────────────────────────────────────────────────
# Splits on the literal "SUMMARY OF IS" header boundary
CHUNK_SPLIT_PATTERN = re.compile(r"(?=SUMMARY\s+OF\s+IS\s+)", re.IGNORECASE)

# Extracts a standard ID like "IS 269 : 1989" or "IS 269:1989" or "IS 269"
STANDARD_ID_PATTERN = re.compile(
    r"IS\s+(\d{1,5})\s*(?::\s*(\d{4}))?", re.IGNORECASE
)


def parse_pdf(pdf_path: Optional[str] = None) -> str:
    """Parse a PDF into markdown using LlamaParse."""
    pdf_path = pdf_path or DATASET_PDF_PATH

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"Dataset PDF not found at {pdf_path}")

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=False,
    )

    logger.info("Parsing PDF via LlamaParse: %s", pdf_path)
    documents = parser.load_data(pdf_path)

    # Combine all pages into a single markdown string
    full_text = "\n\n".join(doc.text for doc in documents)
    logger.info("Parsed %d pages, total %d characters", len(documents), len(full_text))
    return full_text


def regex_chunk(markdown_text: str) -> list[dict]:
    """
    Split the full markdown at every "SUMMARY OF IS" boundary.
    Returns a list of dicts: { "text": ..., "standard_id": ..., "chunk_index": ... }
    """
    raw_chunks = CHUNK_SPLIT_PATTERN.split(markdown_text)

    chunks: list[dict] = []
    for idx, raw in enumerate(raw_chunks):
        text = raw.strip()
        if not text or len(text) < 30:
            # skip trivially small fragments (headers, blanks)
            continue

        # Try to extract a standard ID from the first 300 chars of the chunk
        match = STANDARD_ID_PATTERN.search(text[:300])
        if match:
            std_num = match.group(1)
            std_year = match.group(2)
            standard_id = f"IS {std_num}" + (f": {std_year}" if std_year else "")
        else:
            standard_id = f"UNKNOWN_STD_{idx}"

        chunks.append(
            {
                "text": text,
                "standard_id": standard_id,
                "chunk_index": idx,
            }
        )

    logger.info("Regex chunking produced %d standard chunks", len(chunks))
    return chunks


def build_vectorstore(
    chunks: list[dict],
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> None:
    """Embed chunks and upsert into ChromaDB with metadata."""
    persist_dir = persist_dir or CHROMA_PERSIST_DIR
    collection_name = collection_name or CHROMA_COLLECTION

    # Ensure persistence directory exists
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare batch data
    ids = [f"std_{c['chunk_index']}_{c['standard_id'].replace(' ', '_')}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"standard_id": c["standard_id"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]

    logger.info("Upserting %d chunks into ChromaDB collection '%s'", len(ids), collection_name)
    # Upsert in batches of 100 to avoid memory issues
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    logger.info("Vector store built successfully (%d documents)", collection.count())


def run_ingestion(pdf_path: Optional[str] = None) -> int:
    """
    Full ingestion pipeline: parse → chunk → embed → persist.
    Returns the number of chunks ingested.
    """
    markdown = parse_pdf(pdf_path)
    chunks = regex_chunk(markdown)
    build_vectorstore(chunks)
    return len(chunks)


# ── CLI entrypoint ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    count = run_ingestion()
    print(f"\n✅ Ingestion complete — {count} standard chunks indexed.")
