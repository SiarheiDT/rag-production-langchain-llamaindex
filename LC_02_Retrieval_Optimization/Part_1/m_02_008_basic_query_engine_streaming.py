"""
m_02_008_basic_query_engine_streaming.py

Builds the Paul Graham vector index and runs the basic streaming query engine
example shown in the module.

Usage:
    python m_02_008_basic_query_engine_streaming.py --org-id your-org-id
    python m_02_008_basic_query_engine_streaming.py --org-id your-org-id --question "What does Paul Graham do?"
"""

from __future__ import annotations

import argparse

from m_02_003_common import build_vector_index, require_env

from common.env_loader import load_env

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the basic streaming query engine example.")
    parser.add_argument("--org-id", required=True, help="Activeloop organization ID.")
    parser.add_argument("--dataset-name", default="LlamaIndex_paulgraham_essay", help="Deep Lake dataset name.")
    parser.add_argument("--data-dir", default="./data/paul_graham", help="Directory containing source files.")
    parser.add_argument("--question", default="What does Paul Graham do?", help="Question to ask the query engine.")
    parser.add_argument("--similarity-top-k", type=int, default=10, help="Number of nodes to retrieve.")
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
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=args.similarity_top_k)
    response = query_engine.query(args.question)
    response.print_response_stream()
    print()


if __name__ == "__main__":
    main(parse_args())
