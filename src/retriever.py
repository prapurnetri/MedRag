"""
src/retriever.py
Hybrid retrieval: FAISS (dense) + BM25 (sparse) + cross-encoder re-ranking.
Compatible with langchain 1.x, sentence-transformers 5.x
"""

import sys
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    FAISS_INDEX_DIR, BM25_INDEX_PATH, CHUNKS_PATH,
    EMBEDDING_MODEL, TOP_K_RETRIEVAL, TOP_K_RERANK,
    BM25_WEIGHT, FAISS_WEIGHT
)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


class HybridRetriever:
    """
    3-stage retrieval:
      1. FAISS dense retrieval (semantic similarity)
      2. BM25 sparse retrieval (keyword matching)
      3. Cross-encoder re-ranking (precision boost)
    """

    def __init__(self):
        print("Loading retrieval components...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)

        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        print("Loading re-ranker model...")
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512
        )

        print("Retriever ready.")

    def _faiss_retrieve(self, query: str, k: int) -> list:
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(doc, 1 - float(score)) for doc, score in results]

    def _bm25_retrieve(self, query: str, k: int) -> list:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], scores[i]) for i in top_indices if scores[i] > 0]

    def _merge_results(self, faiss_results: list, bm25_results: list) -> list:
        def normalize(results):
            if not results:
                return []
            scores = [s for _, s in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [(doc, 1.0) for doc, _ in results]
            return [(doc, (s - min_s) / (max_s - min_s)) for doc, s in results]

        faiss_norm = normalize(faiss_results)
        bm25_norm  = normalize(bm25_results)

        combined = {}
        for doc, score in faiss_norm:
            key = doc.page_content[:100]
            combined[key] = {"doc": doc, "score": score * FAISS_WEIGHT}

        for doc, score in bm25_norm:
            key = doc.page_content[:100]
            if key in combined:
                combined[key]["score"] += score * BM25_WEIGHT
            else:
                combined[key] = {"doc": doc, "score": score * BM25_WEIGHT}

        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_results]

    def _rerank(self, query: str, docs: list) -> list:
        if not docs:
            return []
        pairs  = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:TOP_K_RERANK]]

    def retrieve(self, query: str) -> list:
        faiss_results = self._faiss_retrieve(query, TOP_K_RETRIEVAL)
        bm25_results  = self._bm25_retrieve(query, TOP_K_RETRIEVAL)
        merged   = self._merge_results(faiss_results, bm25_results)
        reranked = self._rerank(query, merged)
        return reranked

    def format_context(self, docs: list) -> tuple:
        context_parts = []
        citations     = []

        for i, doc in enumerate(docs, 1):
            title    = doc.metadata.get("title",    "Unknown Paper")
            filename = doc.metadata.get("filename", "unknown.pdf")
            page     = doc.metadata.get("page", 0) + 1

            context_parts.append(
                f"[Source {i}] {title} (p.{page})\n{doc.page_content}"
            )
            citations.append({
                "index":    i,
                "title":    title,
                "filename": filename,
                "page":     page,
                "snippet":  doc.page_content[:200] + "..."
            })

        return "\n\n---\n\n".join(context_parts), citations


if __name__ == "__main__":
    retriever = HybridRetriever()
    query = "What is the role of slow-wave sleep in amyloid-beta clearance?"
    docs  = retriever.retrieve(query)
    _, citations = retriever.format_context(docs)
    print(f"\nRetrieved {len(docs)} chunks")
    for c in citations:
        print(f"  [{c['index']}] {c['title']} — p.{c['page']}")
