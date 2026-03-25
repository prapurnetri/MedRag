"""
src/config.py
All MedRAG settings — compatible with langchain 1.x + Google Gemini 4.x
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
PAPERS_DIR      = BASE_DIR / "data" / "papers"
FAISS_INDEX_DIR = BASE_DIR / "data" / "faiss_index"
BM25_INDEX_PATH = BASE_DIR / "data" / "bm25_index.pkl"
CHUNKS_PATH     = BASE_DIR / "data" / "chunks.pkl"

# ── Groq ─────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
LLM_MODEL       = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0
MAX_TOKENS      = 800

# ── Embedding model (free, runs locally) ─────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 10
TOP_K_RERANK    = 4
BM25_WEIGHT     = 0.3
FAISS_WEIGHT    = 0.7

# ── Hallucination detection ───────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.65

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MedRAG, a clinical research assistant specializing in
Alzheimer's disease and sleep disorders. You answer questions strictly based on
the provided research paper excerpts.

Rules:
1. Only use information from the provided context — never use outside knowledge.
2. Always cite the specific paper and page number for each claim using [Source N].
3. If the context does not contain enough information, say:
   "I don't have enough information in the indexed papers to answer this confidently."
4. Be precise and concise. Use clinical language appropriate for researchers.
5. Structure longer answers with clear numbered points.
"""
