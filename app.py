"""
app.py — Streamlit Demo UI
═══════════════════════════
A minimalist Streamlit interface for live BIS standard recommendations.
Designed for hackathon demo: text input → RAG pipeline → styled JSON output.
"""

import json
import logging
import time

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIS Standard Finder",
    page_icon="🏗️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #00d2ff, #3a7bd5, #6dd5ed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #8892b0;
        font-size: 1.05rem;
    }

    .result-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(12px);
    }

    .standard-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3a7bd5, #00d2ff);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-weight: 500;
        font-size: 0.9rem;
    }

    .latency-tag {
        color: #64ffda;
        font-size: 0.85rem;
        font-weight: 500;
    }

    div[data-testid="stTextArea"] textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #ccd6f6 !important;
        font-size: 1rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3a7bd5, #00d2ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(58, 123, 213, 0.35) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🏗️ BIS Standard Finder</h1>
        <p>AI-powered Bureau of Indian Standards recommendation engine for building materials</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Input ────────────────────────────────────────────────────────────────────
query = st.text_area(
    "Describe your building material or product:",
    placeholder="e.g. Portland pozzolana cement for reinforced concrete construction...",
    height=120,
    key="query_input",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_clicked = st.button("🔍 Find Applicable Standards", use_container_width=True)

# ── Processing ───────────────────────────────────────────────────────────────
if search_clicked and query.strip():
    with st.spinner("🔄 Running hybrid search & reranking..."):
        start = time.time()

        try:
            from src.retriever import retrieve_standards
            from src.generator import generate_response

            chunks = retrieve_standards(query.strip())
            standards = generate_response(query.strip(), chunks)
            latency = time.time() - start

            # Display results
            if standards:
                st.markdown(
                    f'<p class="latency-tag">⚡ Completed in {latency:.2f}s</p>',
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("#### 📋 Applicable BIS Standards")

                badges_html = "".join(
                    f'<span class="standard-badge">{s}</span>' for s in standards
                )
                st.markdown(badges_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # JSON output
                with st.expander("📄 Raw JSON Output", expanded=False):
                    st.json(
                        {
                            "query": query.strip(),
                            "retrieved_standards": standards,
                            "latency_seconds": round(latency, 4),
                            "num_context_chunks": len(chunks),
                        }
                    )

                # Context chunks preview
                with st.expander("🔎 Retrieved Context Chunks", expanded=False):
                    for i, chunk in enumerate(chunks, 1):
                        std_id = chunk.get("metadata", {}).get(
                            "standard_id", "N/A"
                        )
                        score = chunk.get("rerank_score", 0)
                        st.markdown(
                            f"**Chunk {i}** — `{std_id}` (rerank score: {score:.4f})"
                        )
                        st.text(chunk.get("text", "")[:500] + "...")
                        st.divider()
            else:
                st.warning("No applicable standards found for this query.")

        except Exception as e:
            latency = time.time() - start
            st.error(f"Pipeline error: {e}")
            logger.error("Pipeline error: %s", e, exc_info=True)

elif search_clicked:
    st.warning("Please enter a product description first.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#4a5568; font-size:0.85rem; padding:1rem;">
        Built for <strong>BIS x Sigma Squad AI Hackathon</strong> •
        Hybrid RAG Pipeline: ChromaDB + BM25 + Cross-Encoder Reranker •
        Powered by Groq
    </div>
    """,
    unsafe_allow_html=True,
)
