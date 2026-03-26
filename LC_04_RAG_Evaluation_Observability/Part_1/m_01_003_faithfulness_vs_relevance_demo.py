"""
03_faithfulness_vs_relevance_demo.py

A simple educational script that demonstrates the difference between:
- faithfulness
- answer relevance

This script does NOT call an LLM.
Instead, it uses a tiny handcrafted dataset to make the conceptual difference obvious.

Run:
    python 03_faithfulness_vs_relevance_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Example:
    query: str
    context: str
    answer: str
    faithful: bool
    relevant: bool
    explanation: str


EXAMPLES = [
    Example(
        query="What is MRR?",
        context="MRR stands for Mean Reciprocal Rank. It measures how early the first relevant retrieved item appears.",
        answer="MRR stands for Mean Reciprocal Rank. It measures how early the first relevant retrieved item appears.",
        faithful=True,
        relevant=True,
        explanation="The answer is grounded in context and directly answers the query.",
    ),
    Example(
        query="What is MRR?",
        context="MRR stands for Mean Reciprocal Rank. It measures how early the first relevant retrieved item appears.",
        answer="The document also discusses hit rate and NDCG in retrieval systems.",
        faithful=True,
        relevant=False,
        explanation="The answer may be true relative to the context, but it does not answer the actual question.",
    ),
    Example(
        query="What is faithfulness in RAG?",
        context="Faithfulness means the answer should stay grounded in retrieved evidence.",
        answer="Faithfulness is when the system invents plausible extra facts to make the answer more complete.",
        faithful=False,
        relevant=True,
        explanation="The answer discusses the right topic, but the content contradicts the context and is fabricated.",
    ),
]


def main() -> None:
    print("=" * 88)
    print("Faithfulness vs Relevance Demo")
    print("=" * 88)

    for i, item in enumerate(EXAMPLES, start=1):
        print(f"\nExample {i}")
        print(f"Query       : {item.query}")
        print(f"Context     : {item.context}")
        print(f"Answer      : {item.answer}")
        print(f"Faithful?   : {item.faithful}")
        print(f"Relevant?   : {item.relevant}")
        print(f"Explanation : {item.explanation}")

    print("\nTakeaway:")
    print("A response can be faithful but not relevant, and relevant but not faithful.")
    print("In production RAG, you need both.")


if __name__ == "__main__":
    main()
