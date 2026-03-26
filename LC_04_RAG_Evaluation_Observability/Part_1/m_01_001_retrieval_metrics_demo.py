"""
01_retrieval_metrics_demo.py

Educational script for Module 4: RAG Metrics & Evaluation.

What this script does:
- Implements core retrieval metrics from scratch:
  * Hit Rate@k
  * Mean Reciprocal Rank (MRR)
  * Average Precision (AP)
  * Mean Average Precision (MAP)
  * Normalized Discounted Cumulative Gain (NDCG)
- Runs the metrics on a small toy dataset.
- Prints an interview-friendly explanation of the results.

Why it matters:
Retrieval quality is often the first bottleneck in RAG systems.
Even a strong LLM cannot answer correctly if the retriever fails to bring
relevant chunks into the prompt context.

Run:
    python 01_retrieval_metrics_demo.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class RetrievalCase:
    """Represents one query evaluation case.

    Attributes:
        query: User query.
        ranked_doc_ids: Ordered list of retrieved document IDs.
        relevant_doc_ids: Set/list of relevant document IDs.
        graded_relevance: Optional per-document graded relevance used by NDCG.
            If omitted for a doc, relevance defaults to 0.
    """

    query: str
    ranked_doc_ids: list[str]
    relevant_doc_ids: list[str]
    graded_relevance: dict[str, int] | None = None


def hit_rate_at_k(ranked_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str], k: int) -> float:
    """Return 1.0 if any relevant doc appears in top-k, else 0.0."""
    relevant = set(relevant_doc_ids)
    top_k = ranked_doc_ids[:k]
    return 1.0 if any(doc_id in relevant for doc_id in top_k) else 0.0


def reciprocal_rank(ranked_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str]) -> float:
    """Return reciprocal rank of the first relevant result.

    Example:
        first relevant at rank 1 -> 1.0
        first relevant at rank 2 -> 0.5
        no relevant result -> 0.0
    """
    relevant = set(relevant_doc_ids)
    for index, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def average_precision(ranked_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str]) -> float:
    """Compute AP for a single query.

    AP averages precision values at ranks where relevant documents appear.
    """
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0

    num_relevant_found = 0
    precision_sum = 0.0

    for index, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / index
            precision_sum += precision_at_i

    return precision_sum / len(relevant)


def dcg_at_k(ranked_doc_ids: Sequence[str], graded_relevance: dict[str, int], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    score = 0.0
    for index, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = graded_relevance.get(doc_id, 0)
        score += rel / math.log2(index + 1)
    return score


def ndcg_at_k(ranked_doc_ids: Sequence[str], graded_relevance: dict[str, int], k: int) -> float:
    """Compute Normalized DCG at k."""
    actual_dcg = dcg_at_k(ranked_doc_ids, graded_relevance, k)

    ideal_docs = sorted(graded_relevance, key=lambda doc_id: graded_relevance[doc_id], reverse=True)
    ideal_dcg = dcg_at_k(ideal_docs, graded_relevance, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_demo_cases() -> list[RetrievalCase]:
    """Create a tiny dataset with intentionally different retrieval quality."""
    return [
        RetrievalCase(
            query="What is Delta Lake used for?",
            ranked_doc_ids=["d1", "d2", "d3", "d4"],
            relevant_doc_ids=["d1", "d3"],
            graded_relevance={"d1": 3, "d3": 2, "d2": 0, "d4": 0},
        ),
        RetrievalCase(
            query="How does MRR work in retrieval evaluation?",
            ranked_doc_ids=["d8", "d7", "d6", "d5"],
            relevant_doc_ids=["d6"],
            graded_relevance={"d6": 3, "d8": 0, "d7": 0, "d5": 0},
        ),
        RetrievalCase(
            query="What is faithfulness in RAG?",
            ranked_doc_ids=["d10", "d11", "d12"],
            relevant_doc_ids=["d15"],
            graded_relevance={"d10": 0, "d11": 0, "d12": 0, "d15": 3},
        ),
    ]


def main() -> None:
    cases = build_demo_cases()
    k = 3

    hit_rates = []
    reciprocal_ranks = []
    average_precisions = []
    ndcgs = []

    print("=" * 88)
    print("RAG Retrieval Metrics Demo")
    print("=" * 88)

    for i, case in enumerate(cases, start=1):
        hr = hit_rate_at_k(case.ranked_doc_ids, case.relevant_doc_ids, k=k)
        rr = reciprocal_rank(case.ranked_doc_ids, case.relevant_doc_ids)
        ap = average_precision(case.ranked_doc_ids, case.relevant_doc_ids)
        ndcg = ndcg_at_k(case.ranked_doc_ids, case.graded_relevance or {}, k=k)

        hit_rates.append(hr)
        reciprocal_ranks.append(rr)
        average_precisions.append(ap)
        ndcgs.append(ndcg)

        print(f"\nCase {i}")
        print(f"Query              : {case.query}")
        print(f"Retrieved docs     : {case.ranked_doc_ids}")
        print(f"Relevant docs      : {case.relevant_doc_ids}")
        print(f"Hit Rate@{k}       : {hr:.4f}")
        print(f"Reciprocal Rank    : {rr:.4f}")
        print(f"Average Precision  : {ap:.4f}")
        print(f"NDCG@{k}           : {ndcg:.4f}")

    print("\n" + "-" * 88)
    print("Aggregate Metrics")
    print("-" * 88)
    print(f"Mean Hit Rate@{k}  : {mean(hit_rates):.4f}")
    print(f"MRR                : {mean(reciprocal_ranks):.4f}")
    print(f"MAP                : {mean(average_precisions):.4f}")
    print(f"Mean NDCG@{k}      : {mean(ndcgs):.4f}")

    print("\nInterpretation:")
    print("- Hit Rate answers: 'Did we retrieve at least one good chunk in top-k?'")
    print("- MRR answers: 'How early did the first good chunk appear?'")
    print("- MAP answers: 'How good is ranking quality across all relevant chunks?'")
    print("- NDCG answers: 'How well did we rank highly relevant items near the top?'")


if __name__ == "__main__":
    main()
