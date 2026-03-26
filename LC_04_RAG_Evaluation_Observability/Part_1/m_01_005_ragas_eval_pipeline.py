"""
m_01_005_ragas_eval_pipeline.py

Clean RAGAS evaluation script for Module 4 / Part 1.

What this version fixes
-----------------------
1. Uses a schema compatible with newer RAGAS versions:
   - user_input
   - response
   - retrieved_contexts
   - reference
2. Avoids default execution paths that break on embeddings compatibility.
3. Runs stable metrics by default:
   - faithfulness
   - context_precision
   - context_recall
4. Makes answer_relevancy optional because it may fail in some environments
   due to embeddings interface mismatches.
5. Produces cleaner terminal output.

Run
---
python m_01_005_ragas_eval_pipeline.py

Optional
--------
python m_01_005_ragas_eval_pipeline.py --include-answer-relevancy
"""

from __future__ import annotations

import argparse
import sys
import warnings
from common.env_loader import load_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean RAGAS evaluation pipeline")
    parser.add_argument(
        "--include-answer-relevancy",
        action="store_true",
        help="Also run answer_relevancy (may fail in some environments)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env()

    # Keep output cleaner in learning/demo mode
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as exc:
        print("Missing dependency or incompatible package version.")
        print(f"Details: {exc}")
        sys.exit(1)

    rows = {
        "user_input": [
            "What is MRR?",
            "What is faithfulness in RAG?",
        ],
        "response": [
            "MRR is Mean Reciprocal Rank, a metric that measures how early the first relevant result appears.",
            "Faithfulness means the answer is grounded in retrieved context and avoids fabricated claims.",
        ],
        "retrieved_contexts": [
            [
                "MRR stands for Mean Reciprocal Rank and measures the rank position of the first relevant retrieved document."
            ],
            [
                "Faithfulness in RAG checks whether the answer stays supported by retrieved evidence."
            ],
        ],
        "reference": [
            "MRR is Mean Reciprocal Rank.",
            "Faithfulness checks grounding in retrieved context.",
        ],
    }

    dataset = Dataset.from_dict(rows)

    stable_metrics = [faithfulness, context_precision, context_recall]
    metric_names = ["faithfulness", "context_precision", "context_recall"]

    if args.include_answer_relevancy:
        stable_metrics.insert(1, answer_relevancy)
        metric_names.insert(1, "answer_relevancy")

    try:
        result = evaluate(
            dataset=dataset,
            metrics=stable_metrics,
        )

        print("=" * 88)
        print("RAGAS Evaluation (Clean)")
        print("=" * 88)
        print(result)

    except Exception as exc:
        print("=" * 88)
        print("RAGAS Evaluation (Clean)")
        print("=" * 88)

        if args.include_answer_relevancy:
            print("The evaluation failed while answer_relevancy was enabled.")
            print("This usually means there is an embeddings compatibility issue")
            print("in the current environment.")
            print(f"Details: {exc}")
            print()
            print("Try running again without --include-answer-relevancy.")
            sys.exit(1)

        print("Evaluation failed.")
        print(f"Details: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()