# BIS Recommendation Engine
FROM python:3.11-slim

LABEL maintainer="Sigma Squad"
LABEL description="AI-powered BIS Standard Recommendation Engine"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies as a cached layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/chromadb

# Pre-download ML models at build time for faster container cold-start
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# FastAPI: 8000 | Streamlit: 8501
EXPOSE 8000 8501

CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
