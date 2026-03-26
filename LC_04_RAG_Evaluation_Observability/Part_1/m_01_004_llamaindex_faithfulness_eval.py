"""
04_llamaindex_faithfulness_eval.py

Template script for running FaithfulnessEvaluator with LlamaIndex.

This is intentionally written as a clean learning scaffold rather than a
hardwired production script. You can replace the in-memory sample documents
with your own corpus, vector store, or retrieval pipeline.

Requirements (example):
    pip install llama-index llama-index-llms-openai llama-index-embeddings-openai

Environment:
    export OPENAI_API_KEY=...

Run:
    python 04_llamaindex_faithfulness_eval.py
"""

from __future__ import annotations

import os
import sys

from common.env_loader import load_env


def main() -> None:
    load_env()
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.evaluation import FaithfulnessEvaluator
    from llama_index.core.settings import Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    # Configure global settings for a simple educational example.
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    documents = [
        Document(
            text=(
                "Faithfulness in RAG means the answer must stay grounded in the "
                "retrieved context and should not add fabricated claims."
            )
        ),
        Document(
            text=(
                "Answer relevance measures whether the response addresses the "
                "user's question directly and usefully."
            )
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=2)

    query = "What is faithfulness in RAG?"
    response = query_engine.query(query)

    evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    eval_result = evaluator.evaluate_response(response=response)

    print("=" * 88)
    print("LlamaIndex Faithfulness Evaluation")
    print("=" * 88)
    print(f"Query           : {query}")
    print(f"Model Response  : {response}")
    print(f"Passing         : {eval_result.passing}")
    print(f"Feedback        : {getattr(eval_result, 'feedback', None)}")


if __name__ == "__main__":
    main()
