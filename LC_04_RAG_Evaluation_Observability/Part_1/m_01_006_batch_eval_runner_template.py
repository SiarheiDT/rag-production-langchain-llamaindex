"""
06_batch_eval_runner_template.py

Template showing how batch evaluation can be structured for multiple RAG queries.

This script is useful for learning the production mindset:
- do not evaluate one query only
- evaluate batches
- aggregate pass rates
- compare experiments over time

Requirements (example):
    pip install llama-index llama-index-llms-openai llama-index-embeddings-openai

Environment:
    export OPENAI_API_KEY=...

Run:
    python 06_batch_eval_runner_template.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from common.env_loader import load_env

async def async_main() -> None:
    load_env()

    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.evaluation import BatchEvalRunner, FaithfulnessEvaluator, RelevancyEvaluator
    from llama_index.core.settings import Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    documents = [
        Document(text="MRR measures how early the first relevant retrieval result appears."),
        Document(text="Hit Rate checks whether at least one relevant document appears in the top-k results."),
        Document(text="Faithfulness checks whether the answer is supported by retrieved context."),
        Document(text="Answer relevance checks whether the answer addresses the user's query directly."),
    ]

    queries = [
        "What is MRR?",
        "What is Hit Rate?",
        "What is faithfulness in RAG?",
        "What is answer relevance?",
    ]

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=2)

    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)

    runner = BatchEvalRunner(
        evaluators={
            "faithfulness": faithfulness_evaluator,
            "relevancy": relevancy_evaluator,
        },
        workers=4,
    )

    results = await runner.aevaluate_queries(query_engine=query_engine, queries=queries)

    faithfulness_pass_rate = sum(r.passing for r in results["faithfulness"]) / len(results["faithfulness"])
    relevancy_pass_rate = sum(r.passing for r in results["relevancy"]) / len(results["relevancy"])

    print("=" * 88)
    print("Batch Evaluation Summary")
    print("=" * 88)
    print(f"Queries evaluated       : {len(queries)}")
    print(f"Faithfulness pass rate  : {faithfulness_pass_rate:.4f}")
    print(f"Relevancy pass rate     : {relevancy_pass_rate:.4f}")


if __name__ == "__main__":
    asyncio.run(async_main())
