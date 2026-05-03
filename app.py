"""
app.py — Enterprise BIS Compliance Tool (Streamlit UI)
Sleek B2B internal compliance tool for BIS standard recommendations.
Features: system health sidebar, confidence badges, latency metrics.
"""

import json
import logging
import time
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
# Page Config
st.set_page_config(
    page_title="BIS Compliance Engine",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background: #0a0e1a;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.08);
    }
    .sidebar-header {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .health-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.85rem;
    }
    .health-label { color: #94a3b8; }
    .health-value { color: #e2e8f0; font-weight: 500; }
    .health-dot {
        display: inline-block;
        width: 6px; height: 6px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    .dot-green { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
    .dot-blue { background: #3b82f6; box-shadow: 0 0 6px #3b82f6; }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    /* Main Content */
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    /* Result cards */
    .result-container {
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,41,59,0.6));
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        backdrop-filter: blur(20px);
        transition: border-color 0.3s ease;
    }
    .result-container:hover {
        border-color: rgba(56, 189, 248, 0.25);
    }
    .std-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .std-id {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f1f5f9;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-confidence {
        background: rgba(34, 197, 94, 0.12);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    .badge-high {
        background: rgba(34, 197, 94, 0.12);
        color: #22c55e;
    }
    .badge-medium {
        background: rgba(234, 179, 8, 0.12);
        color: #eab308;
    }
    .badge-latency {
        background: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    .badge-agent {
        background: rgba(168, 85, 247, 0.1);
        color: #a855f7;
        border: 1px solid rgba(168, 85, 247, 0.2);
    }
    .std-rationale {
        color: #94a3b8;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-top: 0.3rem;
    }
    .metrics-bar {
        display: flex;
        gap: 0.8rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        text-align: center;
        flex: 1;
        min-width: 120px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }
    /* Input styling */
    div[data-testid="stTextArea"] textarea {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: rgba(56, 189, 248, 0.3) !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.08) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3) !important;
    }
    .divider {
        height: 1px;
        background: rgba(255,255,255,0.06);
        margin: 1.5rem 0;
    }
    .footer {
        text-align: center;
        color: #334155;
        font-size: 0.78rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Sidebar: System Health
with st.sidebar:
    st.markdown("### BIS Engine")
    st.markdown('<p class="sidebar-header">System Health</p>', unsafe_allow_html=True)
    health_items = [
        ("SQLite Cache", "Active (WAL)", "green"),
        ("Vector Store", "ChromaDB", "green"),
        ("Sparse Index", "BM25Okapi", "green"),
        ("Reranker", "ms-marco-MiniLM", "blue"),
        ("Generation", "Groq Llama-3", "blue"),
        ("Architecture", "Dual-Agent", "blue"),
    ]
    for label, value, color in health_items:
        st.markdown(
            f'<div class="health-row">'
            f'<span class="health-label">{label}</span>'
            f'<span class="health-value">'
            f'<span class="health-dot dot-{color}"></span>{value}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        '<p class="sidebar-header" style="margin-top:2rem">Pipeline Config</p>',
        unsafe_allow_html=True,
    )
    config_items = [
        ("Embedding", "all-MiniLM-L6-v2"),
        ("Hybrid Search", "Vector + BM25"),
        ("Fusion", "Reciprocal Rank"),
        ("Verification", "Extract → Verify"),
        ("Latency Target", "< 5 seconds"),
    ]
    for label, value in config_items:
        st.markdown(
            f'<div class="health-row">'
            f'<span class="health-label">{label}</span>'
            f'<span class="health-value">{value}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown(
        '<p style="color:#334155;font-size:0.75rem;text-align:center">'
        "BIS x Sigma Squad · v2.0</p>",
        unsafe_allow_html=True,
    )
# Main Content
st.markdown(
    '<p class="main-title">BIS Standard Compliance Engine</p>'
    '<p class="main-subtitle">'
    "Enter a building material description to identify applicable "
    "Bureau of Indian Standards (SP 21:2005)</p>",
    unsafe_allow_html=True,
)
query = st.text_area(
    "Material or product description:",
    placeholder="e.g. Portland pozzolana cement for reinforced concrete construction...",
    height=100,
    key="query_input",
    label_visibility="collapsed",
)
col1, col2, col3 = st.columns([1, 1.5, 1])
with col2:
    search_clicked = st.button("Analyze Compliance →", use_container_width=True)
# Processing
if search_clicked and query.strip():
    with st.spinner("Running dual-agent RAG pipeline..."):
        start = time.time()
        try:
            from src.retriever import retrieve_standards
            from src.generator import generate_response_detailed
            from inference import QueryCache
            from src.config import SQLITE_CACHE_PATH

            cache = QueryCache(SQLITE_CACHE_PATH)
            cached_report = cache.get(query.strip())

            if cached_report is not None:
                report = cached_report
                chunks = []
            else:
                chunks = retrieve_standards(query.strip())
                report = generate_response_detailed(query.strip(), chunks)
                cache.put(query.strip(), report)

            latency = time.time() - start
            verified = report.get("verified", [])
            dropped = report.get("dropped", [])
            extractor_count = report.get("extractor_count", 0)
            verifier_count = report.get("verifier_count", 0)
            if verified:
                # Metrics bar
                avg_conf = sum(v.get("confidence", 0) for v in verified) / len(verified)
                metrics_html = f"""
                <div class="metrics-bar">
                    <div class="metric-card">
                        <div class="metric-value">{latency:.2f}s</div>
                        <div class="metric-label">Latency</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(verified)}</div>
                        <div class="metric-label">Standards Found</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{avg_conf:.0%}</div>
                        <div class="metric-label">Avg Confidence</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(chunks)}</div>
                        <div class="metric-label">Chunks Retrieved</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(dropped)}</div>
                        <div class="metric-label">Hallucinations Blocked</div>
                    </div>
                </div>
                """
                st.markdown(metrics_html, unsafe_allow_html=True)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                # Standard cards
                for item in verified:
                    std_id = item.get("id", "N/A")
                    rationale = item.get("rationale", "No rationale provided")
                    confidence = item.get("confidence", 0.85)
                    conf_class = "badge-high" if confidence >= 0.8 else "badge-medium"
                    conf_pct = f"{confidence:.0%}"
                    st.markdown(
                        f"""
                        <div class="result-container">
                            <div class="std-header">
                                <span class="std-id">📋 {std_id}</span>
                                <div>
                                    <span class="badge {conf_class}">
                                        ● {conf_pct} confidence
                                    </span>
                                    <span class="badge badge-agent">
                                        ✓ verified
                                    </span>
                                </div>
                            </div>
                            <div class="std-rationale">{rationale}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                # Agent trace
                if dropped:
                    with st.expander("🛡️ Agent Verification Log", expanded=False):
                        st.markdown("**Dropped by Verifier Agent:**")
                        for d in dropped:
                            if isinstance(d, dict):
                                st.markdown(
                                    f"- ~~{d.get('id', 'N/A')}~~ — "
                                    f"*{d.get('reason', 'Failed verification')}*"
                                )
                # Raw JSON
                with st.expander("📄 Raw JSON Output", expanded=False):
                    st.json(
                        {
                            "query": query.strip(),
                            "retrieved_standards": [v["id"] for v in verified],
                            "latency_seconds": round(latency, 4),
                            "agent_report": {
                                "extractor_candidates": extractor_count,
                                "verified_count": verifier_count,
                                "dropped_count": len(dropped),
                            },
                        }
                    )
            else:
                st.warning(
                    "No applicable standards found in the SP 21:2005 database for this query."
                )
        except Exception as e:
            latency = time.time() - start
            st.error(f"Pipeline error: {e}")
            logger.error("Pipeline error: %s", e, exc_info=True)
elif search_clicked:
    st.warning("Please enter a material description.")
# Footer
st.markdown(
    '<div class="footer">'
    "BIS x Sigma Squad AI Hackathon · "
    "Hybrid RAG: ChromaDB + BM25 + Cross-Encoder · "
    "Dual-Agent Verification · Groq Llama-3"
    "</div>",
    unsafe_allow_html=True,
)
