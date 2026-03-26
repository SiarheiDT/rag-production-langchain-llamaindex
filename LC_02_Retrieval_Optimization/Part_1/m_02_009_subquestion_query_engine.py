"""
m_02_009_subquestion_query_engine.py

Builds the Paul Graham vector index and runs the SubQuestionQueryEngine example.

Usage:
    python m_02_009_subquestion_query_engine.py --org-id your-org-id
    python m_02_009_subquestion_query_engine.py --org-id your-org-id --question "How was Paul Graham's life different before, during, and after YC?"
"""

from __future__ import annotations

import argparse

from llama_index.core import Settings
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

from m_02_003_common import build_vector_index, require_env

from common.env_loader import load_env

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SubQuestionQueryEngine example.")
    parser.add_argument("--org-id", required=True, help="Activeloop organization ID.")
    parser.add_argument("--dataset-name", default="LlamaIndex_paulgraham_essay", help="Deep Lake dataset name.")
    parser.add_argument("--data-dir", default="./data/paul_graham", help="Directory containing source files.")
    parser.add_argument(
        "--question",
        default="How was Paul Grahams life different before, during, and after YC?",
        help="Complex question to decompose into sub-questions.",
    )
    parser.add_argument("--similarity-top-k", type=int, default=10, help="Number of nodes to retrieve.")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model used by the sub-question engine.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the dataset before indexing.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    load_env()
    
    require_env("OPENAI_API_KEY")
    index, dataset_path, _ = build_vector_index(
        org_id=args.org_id,
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        overwrite=args.overwrite,
    )

    print(f"Using dataset: {dataset_path}")

    base_query_engine = index.as_query_engine(similarity_top_k=args.similarity_top_k)

    Settings.llm = OpenAI(model=args.model)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=base_query_engine,
            metadata=ToolMetadata(
                name="pg_essay",
                description="Paul Graham essay on What I Worked On",
            ),
        ),
    ]

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        use_async=True,
    )

    response = query_engine.query(args.question)
    print(">>> The final response:\n", response)


if __name__ == "__main__":
    main(parse_args())
