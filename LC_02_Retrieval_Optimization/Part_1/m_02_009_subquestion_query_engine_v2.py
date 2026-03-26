"""
m_02_009_subquestion_query_engine_v2.py

Rewritten SubQuestionQueryEngine example for the current LlamaIndex API.

Key change:
- We DO NOT call SubQuestionQueryEngine.from_defaults(...) without a question generator.
- Instead, we explicitly create LLMQuestionGenerator.from_defaults(...), which avoids
  the optional `llama-index-question-gen-openai` dependency.

Usage:
    export PYTHONPATH=/home/siarhei/rag-production-langchain-llamaindex

    python LC_02_Retrieval_Optimization/Part_1/m_02_009_subquestion_query_engine_v2.py \
      --org-id siarhei \
      --dataset-name pg_essay \
      --data-dir LC_02_Retrieval_Optimization/Part_1/data/paul_graham

    python LC_02_Retrieval_Optimization/Part_1/m_02_009_subquestion_query_engine_v2.py \
      --org-id siarhei \
      --dataset-name pg_essay \
      --data-dir LC_02_Retrieval_Optimization/Part_1/data/paul_graham \
      --question "How was Paul Grahams life different before, during, and after YC?"
"""

from __future__ import annotations

import argparse

from llama_index.core import Settings
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

from m_02_003_common import build_vector_index, require_env
from common.env_loader import load_env

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the sub-question query engine example."""
    parser = argparse.ArgumentParser(
        description="Run the SubQuestionQueryEngine example without llama-index-question-gen-openai."
    )
    parser.add_argument("--org-id", required=True, help="Activeloop organization ID.")
    parser.add_argument(
        "--dataset-name",
        default="LlamaIndex_paulgraham_essay",
        help="Deep Lake dataset name.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data/paul_graham",
        help="Directory containing source files.",
    )
    parser.add_argument(
        "--question",
        default="How was Paul Grahams life different before, during, and after YC?",
        help="Complex question to decompose into sub-questions.",
    )
    parser.add_argument(
        "--similarity-top-k",
        type=int,
        default=10,
        help="Number of nodes to retrieve in the base query engine.",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI model used for question decomposition and synthesis.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the dataset before indexing.",
    )
    parser.add_argument(
        "--use-async",
        action="store_true",
        help="Execute sub-questions asynchronously.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    load_env()
    """Build/load the index and run a sub-question query."""
    require_env("OPENAI_API_KEY")
    require_env("ACTIVELOOP_TOKEN")

    # Build or reuse the vector index.
    index, dataset_path, _ = build_vector_index(
        org_id=args.org_id,
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        overwrite=args.overwrite,
    )

    print(f"Using dataset: {dataset_path}")

    # Base query engine that will answer each generated sub-question.
    base_query_engine = index.as_query_engine(similarity_top_k=args.similarity_top_k)

    # Configure the global LLM used by the engine stack.
    llm = OpenAI(model=args.model)
    Settings.llm = llm

    # Register the base query engine as a tool.
    query_engine_tools = [
        QueryEngineTool(
            query_engine=base_query_engine,
            metadata=ToolMetadata(
                name="pg_essay",
                description="Paul Graham essay on What I Worked On.",
            ),
        ),
    ]

    # IMPORTANT:
    # Explicitly use the prompt-based question generator from llama_index.core.
    # This avoids the optional `llama-index-question-gen-openai` package that
    # caused the dependency conflict in your environment.
    question_gen = LLMQuestionGenerator.from_defaults(llm=llm)

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        llm=llm,
        question_gen=question_gen,
        use_async=args.use_async,
        verbose=True,
    )

    response = query_engine.query(args.question)
    print(">>> The final response:\n")
    print(response)


if __name__ == "__main__":
    main(parse_args())
