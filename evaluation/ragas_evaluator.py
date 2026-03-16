"""
RAGAS evaluation of the Investor agent pipeline.

Runs golden dataset questions through the agent graph and measures:
  - faithfulness: Are claims grounded in retrieved context?
  - answer_relevancy: Does the answer address the question?
  - context_precision: Are retrieved chunks relevant?
  - context_recall: Does retrieved context cover the ground truth?

Usage:
    python -m evaluation.ragas_evaluator
    python -m evaluation.ragas_evaluator --limit 20 --output results.json
"""

import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from agents.graph import graph
from agents.state import initial_state
from evaluation.golden_dataset_generator import load_golden_dataset

DEFAULT_OUTPUT = Path("evaluation/ragas_results.json")


def _run_query(question: str) -> tuple[str, list[str]]:
    """
    Run a question through the agent graph.

    Returns:
        (answer, contexts) where contexts is a list of retrieved chunk texts.
    """
    state = initial_state(user_query=question)
    result = graph.invoke(state)

    answer = result.get("final_response", "")
    reranked = result.get("reranked_chunks", [])
    contexts = [chunk.get("text", "") for chunk in reranked if chunk.get("text")]

    return answer, contexts


def evaluate_pipeline(
    dataset: list[dict] | None = None,
    limit: int | None = None,
    output_file: Path = DEFAULT_OUTPUT,
) -> dict:
    """
    Run RAGAS evaluation on the golden dataset.

    Args:
        dataset: List of Q&A dicts. If None, loads from default path.
        limit: Evaluate only the first N items (for quick testing).
        output_file: Where to save the results JSON.

    Returns:
        Dict with RAGAS metric scores.
    """
    if dataset is None:
        dataset = load_golden_dataset()

    if not dataset:
        raise ValueError("No golden dataset found. Run golden_dataset_generator.py first.")

    if limit:
        dataset = dataset[:limit]

    print(f"Evaluating {len(dataset)} questions...")

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i+1}/{len(dataset)}] {question[:80]}...")

        try:
            answer, contexts = _run_query(question)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            answer = ""
            contexts = item.get("contexts", [])

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts if contexts else item.get("contexts", []))
        ground_truths.append(ground_truth)

    # Build RAGAS dataset
    ragas_data = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    # Run evaluation
    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        ragas_data,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    scores = {
        "faithfulness": float(results["faithfulness"]),
        "answer_relevancy": float(results["answer_relevancy"]),
        "context_precision": float(results["context_precision"]),
        "context_recall": float(results["context_recall"]),
        "n_questions": len(questions),
    }

    print("\n=== RAGAS Results ===")
    for metric, score in scores.items():
        if isinstance(score, float):
            print(f"  {metric}: {score:.4f}")

    # Save results
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nResults saved → {output_file}")

    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    evaluate_pipeline(limit=args.limit, output_file=args.output)
