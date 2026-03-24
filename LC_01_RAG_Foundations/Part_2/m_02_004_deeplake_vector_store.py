"""
Example 4: Store LlamaIndex documents in Deep Lake.

This script follows the lecture example that creates a Deep Lake
vector store and writes document embeddings to it.

Requirements:
    - ACTIVELOOP_TOKEN
    - OPENAI_API_KEY

Usage:
    python m_02_004_deeplake_vector_store.py --org_id your_org_id --dataset_name LlamaIndex_intro
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.env_loader import load_env
from llama_index.core import StorageContext, VectorStoreIndex, download_loader
from llama_index.vector_stores.deeplake import DeepLakeVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and populate a Deep Lake vector store.")
    parser.add_argument("--org_id", type=str, required=True, help="ActiveLoop organization ID.")
    parser.add_argument("--dataset_name", type=str, default="LlamaIndex_intro", help="Deep Lake dataset name.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the dataset if it already exists.")
    parser.add_argument(
        "--pages",
        nargs="+",
        default=["Natural Language Processing", "Artificial Intelligence"],
        help="List of Wikipedia page titles to load.",
    )
    return parser.parse_args()


def main() -> None:
    load_env()
    args = parse_args()

    dataset_path = f"hub://{args.org_id}/{args.dataset_name}"

    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    documents = loader.load_data(pages=args.pages)

    vector_store = DeepLakeVectorStore(
        dataset_path=dataset_path,
        overwrite=args.overwrite,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    print("Deep Lake dataset path:")
    print(dataset_path)
    print("\nIndex created successfully:")
    print(index)


if __name__ == "__main__":
    main()
