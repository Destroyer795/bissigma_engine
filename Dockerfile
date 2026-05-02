# ═══════════════════════════════════════════════════════════════════════════
# Dockerfile — BIS Recommendation Engine (Local Development)
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim

LABEL maintainer="Sigma Squad"
LABEL description="AI-powered BIS Standard Recommendation Engine"

# ── System dependencies ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (cached layer) ──────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ───────────────────────────────────────────────
COPY . .

# ── Create data directory ───────────────────────────────────────────────
RUN mkdir -p /app/data/chromadb

# ── Expose ports ─────────────────────────────────────────────────────────
# FastAPI: 8000 | Streamlit: 8501
EXPOSE 8000 8501

# ── Default: run FastAPI backend ─────────────────────────────────────────
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
