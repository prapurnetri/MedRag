"""
evaluation/evaluate.py
Simple MedRAG evaluation using built-in faithfulness scoring.
No RAGAS dependency — works with any LLM backend.
"""

import sys
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import MedRAGPipeline


TEST_QA_PAIRS = [
    {
        "question": "What is the role of slow-wave sleep in amyloid-beta clearance?",
        "ground_truth": "Slow-wave sleep facilitates glymphatic clearance of amyloid-beta through cerebrospinal fluid flow."
    },
    {
        "question": "What cognitive tests are used to assess MCI conversion risk?",
        "ground_truth": "MMSE, MoCA, CDR, and ADAS-Cog are common cognitive assessments for MCI conversion risk."
    },
    {
        "question": "How does APOE4 genotype affect Alzheimer's risk?",
        "ground_truth": "APOE4 is the strongest genetic risk factor for late-onset Alzheimer's, increasing risk 3-4x."
    },
    {
        "question": "What neuroimaging biomarkers predict MCI-to-AD conversion?",
        "ground_truth": "Hippocampal atrophy on MRI and amyloid deposition on PET are key predictive biomarkers."
    },
    {
        "question": "What is the glymphatic system and when is it most active?",
        "ground_truth": "The glymphatic system clears brain waste via CSF flow and is most active during slow-wave sleep."
    },
    {
        "question": "How do deep learning models improve Alzheimer's detection accuracy?",
        "ground_truth": "CNNs on MRI data achieve AUC 0.80-0.95 for AD detection, outperforming traditional ML."
    },
    {
        "question": "What sleep disturbances are most common in Alzheimer's patients?",
        "ground_truth": "Insomnia, reduced slow-wave sleep, disrupted circadian rhythms, and sundowning are common."
    },
    {
        "question": "What is the role of tau protein in Alzheimer's disease progression?",
        "ground_truth": "Hyperphosphorylated tau forms neurofibrillary tangles and spreads between neurons, correlating with cognitive decline."
    },
    {
        "question": "How does multimodal fusion improve MCI conversion prediction?",
        "ground_truth": "Combining MRI with clinical and cognitive features improves AUC beyond single-modality approaches."
    },
    {
        "question": "What datasets are commonly used for Alzheimer's deep learning research?",
        "ground_truth": "ADNI is the primary dataset, providing MRI, PET, cognitive scores, and genetic data for AD research."
    },
]


def run_evaluation(pipeline: MedRAGPipeline, qa_pairs: list) -> dict:
    print(f"\nRunning evaluation on {len(qa_pairs)} Q&A pairs...")
    print("=" * 60)

    faithfulness_scores = []
    results_detail      = []

    for i, pair in enumerate(qa_pairs):
        print(f"\nQ{i+1}: {pair['question'][:65]}...")
        response = pipeline.ask(pair["question"], include_vanilla=False)

        faithfulness_scores.append(response.faithfulness)
        results_detail.append({
            "question":    pair["question"],
            "answer":      response.answer,
            "faithfulness": response.faithfulness,
            "sources":     len(response.citations),
            "hallucination_flag": response.is_hallucinating,
        })

        print(f"     Faithfulness: {response.faithfulness:.2f} | "
              f"Sources cited: {len(response.citations)} | "
              f"Hallucination: {'YES' if response.is_hallucinating else 'No'}")

    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    flagged          = sum(1 for r in results_detail if r["hallucination_flag"])
    avg_sources      = sum(r["sources"] for r in results_detail) / len(results_detail)

    return {
        "faithfulness":     avg_faithfulness,
        "flagged_answers":  flagged,
        "total_questions":  len(qa_pairs),
        "avg_sources_cited": avg_sources,
        "detail":           results_detail,
    }


def print_summary(results: dict):
    print("\n" + "=" * 60)
    print("MEDRAG EVALUATION RESULTS")
    print("=" * 60)
    print(f"Questions evaluated:   {results['total_questions']}")
    print(f"Avg faithfulness:      {results['faithfulness']:.2f}  (target: > 0.80)")
    print(f"Avg sources cited:     {results['avg_sources_cited']:.1f}  per answer")
    print(f"Hallucination flags:   {results['flagged_answers']} / {results['total_questions']}")
    print("=" * 60)
    print("\nUse these scores in your:")
    print(f"  Resume:  'Achieved {results['faithfulness']:.2f} faithfulness score across 10 clinical Q&A pairs'")
    print(f"  README:  Faithfulness: {results['faithfulness']:.2f} | Avg citations: {results['avg_sources_cited']:.1f}/answer")


def save_results(results: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Faithfulness", "Sources Cited", "Hallucination Flag", "Answer Preview"])
        for r in results["detail"]:
            writer.writerow([
                r["question"],
                f"{r['faithfulness']:.2f}",
                r["sources"],
                "YES" if r["hallucination_flag"] else "No",
                r["answer"][:100] + "...",
            ])

    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    pipeline = MedRAGPipeline()
    results  = run_evaluation(pipeline, TEST_QA_PAIRS)
    print_summary(results)
    save_results(results, Path("evaluation/results.csv"))
