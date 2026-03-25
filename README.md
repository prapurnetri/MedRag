# MedRAG — Clinical Research Q&A with Hallucination Detection

A production-quality RAG pipeline over Alzheimer's and sleep disorder research papers.
Built with LangChain, FAISS hybrid search, GPT-4o, and RAGAS evaluation.

## Features
- Hybrid search (dense FAISS + sparse BM25) over 100+ clinical papers
- Citation grounding — every answer traces back to a source paper + page
- Hallucination detection via RAGAS faithfulness scoring
- Automatic warning flag when answer faithfulness drops below threshold
- Side-by-side comparison: MedRAG vs vanilla GPT-4 (no context)
- Streamlit UI deployable on HuggingFace Spaces

## Project Structure
```
medrag/
├── data/
│   └── papers/          ← Put your PDF papers here
├── src/
│   ├── ingest.py        ← PDF loading, chunking, embedding, FAISS index
│   ├── retriever.py     ← Hybrid BM25 + FAISS retrieval with re-ranking
│   ├── pipeline.py      ← Full RAG chain: retrieve → generate → cite → flag
│   └── config.py        ← All settings in one place
├── evaluation/
│   └── evaluate.py      ← RAGAS evaluation: faithfulness, relevance, recall
├── app/
│   └── streamlit_app.py ← Full Streamlit UI with citation + hallucination flag
├── requirements.txt
└── README.md
```

## Setup (Step by Step)

### Step 1 — Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/medrag.git
cd medrag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Add your OpenAI API key
```bash
cp .env.example .env
# Open .env and paste your OpenAI API key
```
Get a free API key at: https://platform.openai.com/api-keys
New accounts get $5 free credit — more than enough for this project.

### Step 3 — Collect papers (80–100 PDFs)
Download PDFs from:
- PubMed: https://pubmed.ncbi.nlm.nih.gov (search "Alzheimer's MCI conversion")
- arXiv: https://arxiv.org (search "Alzheimer's deep learning 2024")
- Semantic Scholar: https://www.semanticscholar.org

Save all PDFs to: `data/papers/`

### Step 4 — Ingest papers (build the vector index)
```bash
python src/ingest.py
```
This will:
- Load all PDFs from data/papers/
- Split into chunks of 512 tokens with 50-token overlap
- Embed using sentence-transformers
- Save FAISS index to data/faiss_index/
- Save BM25 index to data/bm25_index.pkl

Takes about 5–10 minutes for 100 papers.

### Step 5 — Test the pipeline
```bash
python src/pipeline.py
```
Runs 3 sample questions and prints answers with citations.

### Step 6 — Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```
Opens at http://localhost:8501

### Step 7 — Run RAGAS evaluation
```bash
python evaluation/evaluate.py
```
Prints faithfulness, answer relevance, and context recall scores.

## Deployment (HuggingFace Spaces — free)
1. Create account at huggingface.co
2. New Space → Streamlit → upload your files
3. Add OPENAI_API_KEY in Space Settings → Secrets
4. Your demo is live at: huggingface.co/spaces/YOUR_USERNAME/medrag

## Resume Bullet (fill in your scores after running evaluation)
"Built MedRAG, a production RAG pipeline over 100+ Alzheimer's and sleep disorder
research papers using LangChain, FAISS hybrid search, and GPT-4o. Implemented
citation grounding and hallucination detection via RAGAS (faithfulness: X.XX).
Deployed on HuggingFace Spaces with Streamlit. Achieved X% improvement in answer
faithfulness over vanilla LLM baseline."
