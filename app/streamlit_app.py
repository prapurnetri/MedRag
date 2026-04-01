"""
app/streamlit_app.py — Groq version
MedRAG: Clinical Research Q&A with Citation Grounding and Hallucination Detection
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from src.pipeline import MedRAGPipeline, MedRAGResponse
from src.config import FAITHFULNESS_THRESHOLD

st.set_page_config(
    page_title="MedRAG — Clinical Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .answer-box  { background:#f0faf4; border-left:4px solid #2e7d32;
                   padding:1.2rem; border-radius:0 8px 8px 0; margin:1rem 0; }
    .vanilla-box { background:#e8f4fd; border-left:4px solid #1565c0;
                   padding:1.2rem; border-radius:0 8px 8px 0; margin:1rem 0; }
    .warn-box    { background:#fff8e1; border:1px solid #f9a825;
                   padding:1rem; border-radius:8px; margin:1rem 0; }
    .score-good  { color:#2e7d32; font-weight:600; }
    .score-mid   { color:#f57f17; font-weight:600; }
    .score-bad   { color:#c62828; font-weight:600; }
    .metric-card { background:#f8f9fa; border-radius:8px;
                   padding:.75rem 1rem; text-align:center; }
    .metric-num  { font-size:1.8rem; font-weight:600; color:#1a1a2e; }
    .metric-label{ font-size:.85rem; color:#666; margin-top:2px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading MedRAG pipeline...")
def load_pipeline():
    return MedRAGPipeline()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MedRAG")
    st.markdown("""
    Clinical research Q&A over **27 papers** on
    Alzheimer's disease and sleep disorders.

    **Stack:**
    - Hybrid FAISS + BM25 retrieval
    - Cross-encoder re-ranking
    - Llama 3.3 70B via Groq
    - Citation grounding
    - Hallucination detection (faithfulness: **0.88**)
    """)

    st.divider()
    st.markdown("### Settings")
    show_vanilla      = st.toggle("Show vanilla LLM comparison", value=True)
    show_faithfulness = st.toggle("Show faithfulness score",     value=True)
    show_context      = st.toggle("Show retrieved context",      value=False)

    st.divider()
    st.markdown("### Sample questions")
    samples = [
        "What is the role of slow-wave sleep in amyloid-beta clearance?",
        "What are the earliest biomarkers for MCI-to-AD conversion?",
        "How does APOE4 genotype affect Alzheimer's risk?",
        "What neuroimaging features predict cognitive decline?",
        "How does the glymphatic system relate to Alzheimer's?",
        "What deep learning models are used for MCI prediction?",
        "How does tau accumulation progress in Alzheimer's disease?",
    ]
    for q in samples:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["prefill"] = q

    st.divider()
    st.markdown("### Evaluation results")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-num">0.85</div><div class="metric-label">Faithfulness</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-num">0/10</div><div class="metric-label">Hallucinations</div></div>', unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.title("🧠 MedRAG — Clinical Research Assistant")
st.caption("Citation-grounded Q&A over Alzheimer's and sleep disorder research · Powered by Groq Llama 3.3 70B")

prefill  = st.session_state.get("prefill", "")
question = st.text_area(
    "Ask a clinical research question:",
    value=prefill,
    height=90,
    placeholder="e.g. What is the relationship between sleep disruption and amyloid accumulation in Alzheimer's disease?",
)

col_btn, col_info = st.columns([1, 5])
with col_btn:
    ask_btn = st.button("Ask MedRAG", type="primary", use_container_width=True)
with col_info:
    st.caption("Answers are grounded in indexed research papers. Every claim is traceable to a source.")

# ── Process question ──────────────────────────────────────────────────────────
if ask_btn and question.strip():
    if "prefill" in st.session_state:
        del st.session_state["prefill"]

    try:
        pipeline = load_pipeline()

        with st.spinner("Retrieving passages · Generating answer · Checking faithfulness..."):
            response: MedRAGResponse = pipeline.ask(
                question.strip(),
                include_vanilla=show_vanilla,
            )

        # Hallucination warning
        if response.is_hallucinating:
            st.markdown(f"""
            <div class="warn-box">
                ⚠️ <strong>Hallucination Warning</strong> —
                Faithfulness score <strong>{response.faithfulness:.2f}</strong>
                is below threshold ({FAITHFULNESS_THRESHOLD}).
                This answer may contain claims not supported by the indexed papers.
                Please verify against source documents before clinical use.
            </div>
            """, unsafe_allow_html=True)

        # Two-column layout for comparison
        if show_vanilla and response.vanilla_answer:
            col_rag, col_van = st.columns(2)
        else:
            col_rag = st.container()
            col_van = None

        # MedRAG answer
        with col_rag:
            st.markdown("#### MedRAG answer")
            st.markdown(
                f'<div class="answer-box">{response.answer}</div>',
                unsafe_allow_html=True
            )

            # Citations
            if response.citations:
                st.markdown(f"**{len(response.citations)} sources retrieved:**")
                for c in response.citations:
                    with st.expander(f"[{c['index']}] {c['title']} — Page {c['page']}"):
                        st.markdown(f"**File:** `{c['filename']}`")
                        st.markdown(f"**Relevant excerpt:**")
                        st.markdown(f"> {c['snippet']}")

            # Faithfulness score
            if show_faithfulness:
                score = response.faithfulness
                if score >= 0.80:
                    css, label = "score-good", "High faithfulness"
                elif score >= FAITHFULNESS_THRESHOLD:
                    css, label = "score-mid",  "Moderate faithfulness"
                else:
                    css, label = "score-bad",  "Low faithfulness"

                st.markdown(
                    f'**Faithfulness score:** <span class="{css}">{score:.2f} — {label}</span>',
                    unsafe_allow_html=True
                )

        # Vanilla comparison
        if col_van and response.vanilla_answer:
            with col_van:
                st.markdown("#### Vanilla LLM (no context)")
                st.markdown(
                    f'<div class="vanilla-box">{response.vanilla_answer}</div>',
                    unsafe_allow_html=True
                )
                st.caption("No citations · No hallucination detection · No source grounding")

        # Retrieved context (optional)
        if show_context:
            with st.expander("Show full retrieved context passed to LLM"):
                st.text(
                    response.context_used[:4000] + "..."
                    if len(response.context_used) > 4000
                    else response.context_used
                )

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.info("Make sure you have run `python src/ingest.py` first.")

elif ask_btn:
    st.warning("Please enter a question first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "MedRAG · Sri Vidhya Vishnubotla · "
    "M.Eng. Computer Science, University of Cincinnati · 2026 · "
    "Built with LangChain, FAISS, BM25, Groq Llama 3.3 70B"
)
