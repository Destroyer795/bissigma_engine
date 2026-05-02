# 🏗️ BIS Standard Recommendation Engine

> **BIS x Sigma Squad AI Hackathon** — AI-powered recommendation engine that maps building material descriptions to applicable Bureau of Indian Standards (BIS).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      inference.py (Judge Entry)                 │
│                   SQLite WAL Cache (sub-ms lookups)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │  ChromaDB    │   │    BM25      │   │  Cross-Encoder   │    │
│  │  (Dense)     │──▶│  (Sparse)    │──▶│  Reranker        │    │
│  │  Vector      │   │  Keyword     │   │  BAAI/bge-base   │    │
│  └──────────────┘   └──────────────┘   └──────────────────┘    │
│         │                                       │               │
│         └───────── RRF Fusion ──────────────────┘               │
│                         │                                       │
│                   ┌─────▼──────┐                                │
│                   │  Groq LLM  │                                │
│                   │  (llama3)  │                                │
│                   └────────────┘                                │
├─────────────────────────────────────────────────────────────────┤
│  LlamaParse (PDF → MD) → Regex Chunker ("SUMMARY OF IS")       │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Setup

```bash
# Clone and enter project
cd bissigma_engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your GROQ_API_KEY and LLAMA_CLOUD_API_KEY
```

### 2. Data Ingestion

Place your `dataset.pdf` (SP 21:2005) in the `data/` folder, then:

```bash
python -m src.ingest
```

### 3. Run Inference (Judge's Script)

```bash
python inference.py --input data/queries.json --output data/results.json
```

### 4. Launch Demo UI

```bash
streamlit run app.py
```

### 5. Docker (Full Stack)

```bash
docker-compose up --build
```
- **API**: http://localhost:8000/docs
- **UI**: http://localhost:8501

## Project Structure

```
├── src/
│   ├── __init__.py          # Package init
│   ├── config.py            # Environment configuration
│   ├── ingest.py            # LlamaParse + Regex chunking pipeline
│   ├── retriever.py         # Hybrid search (Vector + BM25) + Reranking
│   └── generator.py         # Groq LLM generation
├── data/                    # PDFs, ChromaDB, SQLite cache
├── inference.py             # Judge's bulletproof entry point
├── app.py                   # Streamlit demo UI
├── app_api.py               # FastAPI microservice backend
├── Dockerfile               # Container image
├── docker-compose.yml       # Local dev stack
└── requirements.txt         # Python dependencies
```

## Key Design Decisions

| Concern | Solution |
|---------|----------|
| **Zero-crash** | Every query wrapped in `try/except` → empty `[]` fallback |
| **Latency < 5s** | SQLite WAL cache, Groq ultra-fast inference, local ChromaDB |
| **Accuracy >80%** | Hybrid search (dense + sparse) + Cross-Encoder reranker |
| **Chunking** | Regex split at "SUMMARY OF IS" — one chunk = one standard |
| **Compliance** | Strict system prompt prevents LLM hallucination |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/recommend` | Get BIS recommendations for a query |
| `POST` | `/ingest` | Trigger data ingestion pipeline |
| `GET` | `/docs` | Interactive Swagger UI |

---

*Built with ❤️ by Sigma Squad*
