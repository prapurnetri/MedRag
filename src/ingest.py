"""
src/ingest.py
Loads PDFs, chunks them, builds FAISS + BM25 indexes.
Compatible with langchain 1.x, sentence-transformers 5.x, faiss-cpu 1.13.x

Run once: python src/ingest.py
Re-run whenever you add new papers.
"""

import sys
import pickle
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    PAPERS_DIR, FAISS_INDEX_DIR, BM25_INDEX_PATH,
    CHUNKS_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


def load_pdfs(papers_dir: Path) -> list:
    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {papers_dir}")
        print("Add your PDFs to data/papers/ and re-run.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDFs. Loading...")
    all_docs = []

    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["filename"] = pdf_path.name
                doc.metadata["title"]    = pdf_path.stem.replace("_", " ").replace("-", " ")
            all_docs.extend(docs)
        except Exception as e:
            print(f"Warning: Could not load {pdf_path.name}: {e}")

    print(f"Loaded {len(all_docs)} pages from {len(pdf_files)} papers.")
    return all_docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    return chunks


def build_faiss_index(chunks: list) -> FAISS:
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads ~90MB — subsequent runs are instant)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def build_bm25_index(chunks: list) -> BM25Okapi:
    print("Building BM25 index...")
    tokenized = [chunk.page_content.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


def save_artifacts(vectorstore, bm25, chunks):
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))
    print(f"FAISS index saved to {FAISS_INDEX_DIR}")

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved to {BM25_INDEX_PATH}")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {CHUNKS_PATH}")


def main():
    print("=" * 60)
    print("MedRAG — Ingestion Pipeline")
    print("=" * 60)

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    docs   = load_pdfs(PAPERS_DIR)
    chunks = chunk_documents(docs)
    vectorstore = build_faiss_index(chunks)
    bm25  = build_bm25_index(chunks)
    save_artifacts(vectorstore, bm25, chunks)

    print("\nIngestion complete!")
    print(f"Indexed {len(chunks)} chunks from your papers.")
    print("\nNext step: python src/pipeline.py")


if __name__ == "__main__":
    main()
