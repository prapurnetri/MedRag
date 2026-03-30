"""
src/pipeline.py
Full MedRAG pipeline — using Groq (free, fast, no quota issues)
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    MAX_TOKENS, SYSTEM_PROMPT, FAITHFULNESS_THRESHOLD
)
from src.retriever import HybridRetriever

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


@dataclass
class MedRAGResponse:
    question:         str
    answer:           str
    citations:        list
    faithfulness:     float
    is_hallucinating: bool
    context_used:     str
    vanilla_answer:   str = ""


class MedRAGPipeline:

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "1. Open your .env file\n"
                "2. Add: GROQ_API_KEY=gsk_your-key-here\n"
                "3. Get free key at: https://console.groq.com"
            )

        self.retriever = HybridRetriever()

        self.llm = ChatGroq(
            model=LLM_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        print("MedRAG pipeline ready (Groq Llama 3.3 70B).")

    def _generate_answer(self, question: str, context: str) -> str:
        full_prompt = f"""{SYSTEM_PROMPT}

Context from research papers:
{context}

Question: {question}

Answer based strictly on the context above. Cite sources as [Source N]."""

        response = self.llm.invoke([HumanMessage(content=full_prompt)])
        return response.content

    def _generate_vanilla_answer(self, question: str) -> str:
        prompt = f"You are a helpful medical AI assistant. Answer this question:\n\n{question}"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _faithfulness_check(self, answer: str, context: str) -> float:
        prompt = f"""You are a strict evaluator. Score ONLY based on evidence present in the context.

Context from research papers:
{context[:2000]}

AI Answer to evaluate:
{answer}

Instructions:
1. Read each claim in the answer carefully.
2. Check whether each claim is directly supported or paraphrased from the context above.
3. Count how many claims are supported vs invented from outside knowledge.

Scoring guide:
- 1.0 = every single claim is found in the context
- 0.9 = almost all claims supported, tiny extrapolation
- 0.8 = most claims supported, minor outside knowledge
- 0.7 = majority supported but some claims not in context
- 0.6 = roughly half supported, half invented
- 0.4 = few claims supported, mostly outside knowledge
- 0.2 = almost no claims from context
- 0.0 = answer completely ignores the context

Be strict. Do not give 0.85 by default. Evaluate carefully and give a specific score.
Respond with ONLY a decimal number. Examples: 0.92, 0.74, 0.61, 0.88"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            match = re.search(r"0\.\d+|1\.0", response.content.strip())
            score = float(match.group()) if match else 0.5
            return min(max(score, 0.0), 1.0)
        except Exception:
            return 0.5

    def ask(self, question: str, include_vanilla: bool = False) -> MedRAGResponse:
        print(f"\nQuestion: {question}")

        print("Retrieving relevant passages...")
        docs = self.retriever.retrieve(question)
        context, citations = self.retriever.format_context(docs)

        print("Generating answer...")
        answer = self._generate_answer(question, context)

        print("Checking faithfulness...")
        faithfulness = self._faithfulness_check(answer, context)
        is_hallucinating = faithfulness < FAITHFULNESS_THRESHOLD

        vanilla_answer = ""
        if include_vanilla:
            print("Generating vanilla comparison...")
            vanilla_answer = self._generate_vanilla_answer(question)

        print(f"Faithfulness: {faithfulness:.2f}")
        if is_hallucinating:
            print("WARNING: Low faithfulness detected")

        return MedRAGResponse(
            question=question,
            answer=answer,
            citations=citations,
            faithfulness=faithfulness,
            is_hallucinating=is_hallucinating,
            context_used=context,
            vanilla_answer=vanilla_answer,
        )


def print_response(r: MedRAGResponse):
    print("\n" + "=" * 70)
    print(f"QUESTION: {r.question}")
    print("=" * 70)
    print(f"\nANSWER:\n{r.answer}")
    print(f"\nFAITHFULNESS: {r.faithfulness:.2f}")
    if r.is_hallucinating:
        print("HALLUCINATION WARNING: Verify this answer carefully.")
    print(f"\nSOURCES ({len(r.citations)}):")
    for c in r.citations:
        print(f"  [{c['index']}] {c['title']} — Page {c['page']}")
    if r.vanilla_answer:
        print("\nVANILLA GROQ (no context):")
        print(r.vanilla_answer[:400] + "..." if len(r.vanilla_answer) > 400 else r.vanilla_answer)


if __name__ == "__main__":
    pipeline = MedRAGPipeline()
    questions = [
        "What is the role of slow-wave sleep in amyloid-beta clearance?",
        "What are the earliest biomarkers for MCI-to-Alzheimer's conversion?",
        "How does REM sleep disruption relate to tau accumulation?",
    ]
    for q in questions:
        r = pipeline.ask(q, include_vanilla=True)
        print_response(r)