---

title: MedRAG
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: docker
app_file: app/streamlit_app.py
pinned: false

---

# MedRAG — Clinical Research Q&A System

A production-quality Retrieval-Augmented Generation pipeline for clinical research Q&A over Alzheimer's disease and sleep disorder literature.

## Overview

MedRAG enables researchers and clinicians to ask natural language questions over a curated corpus of peer-reviewed papers. Every answer is grounded in source documents with citation tracking and hallucination detection.

## Features

- **Hybrid retrieval** — dense FAISS vector search combined with sparse BM25 keyword search
- **Cross-encoder re-ranking** — precision boost using ms-marco-MiniLM-L-6-v2
- **Citation grounding** — every answer traces back to a specific paper and page number
- **Hallucination detection** — faithfulness scoring flags unsupported claims automatically
- **Side-by-side comparison** — MedRAG vs vanilla LLM to demonstrate grounding value
- **Streamlit interface** — clean UI deployable locally or on HuggingFace Spaces

## Architecture

```
User question
     ↓
Hybrid retrieval (FAISS + BM25)
     ↓
Cross-encoder re-ranking
     ↓
Context formatting with source labels
     ↓
LLM generation (Llama 3.3 70B via Groq)
     ↓
Faithfulness check
     ↓
Answer + citations + hallucination flag
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS |
| Sparse retrieval | BM25 (rank-bm25) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| RAG framework | LangChain 1.x |
| UI | Streamlit |

## Project Structure

```
medrag/
├── data/
│   └── papers/          ← Research PDFs (not tracked)
├── src/
│   ├── config.py        ← Settings and constants
│   ├── ingest.py        ← PDF ingestion and indexing
│   ├── retriever.py     ← Hybrid retrieval pipeline
│   └── pipeline.py      ← End-to-end RAG chain
├── evaluation/
│   └── evaluate.py      ← Faithfulness evaluation
├── app/
│   └── streamlit_app.py ← Web interface
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/prapurnetri/medrag.git
cd medrag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

Get a free Groq API key at: https://console.groq.com

## Usage

**Step 1 — Add papers:**
Place research PDFs in `data/papers/`

**Step 2 — Build index:**
```bash
python src/ingest.py
```

**Step 3 — Run the app:**
```bash
streamlit run app/streamlit_app.py
```

**Step 4 — Evaluate:**
```bash
python evaluation/evaluate.py
```

## Evaluation Results

| Metric | Score |
|---|---|
| Faithfulness | 0.85 |
| Hallucination flags | 0 / 10 |
| Avg sources cited | 4.0 per answer |

## Deployment

Deploy on HuggingFace Spaces (free):
1. Create a new Streamlit Space at huggingface.co
2. Upload project files
3. Add `GROQ_API_KEY` in Space Settings → Secrets

## License

MIT
