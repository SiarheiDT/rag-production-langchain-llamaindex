"""
m_02_011_cohere_rerank_llamaindex.py

Builds the Paul Graham vector index and runs the CohereRerank postprocessor
inside a LlamaIndex query engine.

Usage:
    python m_02_011_cohere_rerank_llamaindex.py --org-id your-org-id
    python m_02_011_cohere_rerank_llamaindex.py --org-id your-org-id --question "What did Sam Altman do in this essay?"
"""

from __future__ import annotations

import argparse
import os

from llama_index.postprocessor.cohere_rerank import CohereRerank

from m_02_003_common import build_vector_index, require_env
from common.env_loader import load_env

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cohere rerank inside a LlamaIndex query engine.")
    parser.add_argument("--org-id", required=True, help="Activeloop organization ID.")
    parser.add_argument("--dataset-name", default="LlamaIndex_paulgraham_essay", help="Deep Lake dataset name.")
    parser.add_argument("--data-dir", default="./data/paul_graham", help="Directory containing source files.")
    parser.add_argument("--question", default="What did Sam Altman do in this essay?", help="Question to ask.")
    parser.add_argument("--similarity-top-k", type=int, default=10, help="Initial retriever top-k.")
    parser.add_argument("--top-n", type=int, default=2, help="Number of documents kept after reranking.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the dataset before indexing.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    load_env()
    require_env("OPENAI_API_KEY")
    require_env("COHERE_API_KEY")

    index, dataset_path, _ = build_vector_index(
        org_id=args.org_id,
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        overwrite=args.overwrite,
    )

    print(f"Using dataset: {dataset_path}")

    cohere_rerank = CohereRerank(api_key=os.environ["COHERE_API_KEY"], top_n=args.top_n)

    query_engine = index.as_query_engine(
        similarity_top_k=args.similarity_top_k,
        node_postprocessors=[cohere_rerank],
    )

    response = query_engine.query(args.question)
    print(response)


if __name__ == "__main__":
    main(parse_args())
