"""
Example 6: Load an existing local index or create a new one.

This script follows the lecture example that:
- checks whether a local storage directory exists
- loads the index if present
- otherwise creates and persists a new one

Usage:
    python m_02_006_load_or_create_index.py --persist_dir ./storage
    python run.py --module 1 --part 2 --task 6 -- --query "What does NLP stand for?"
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.env_loader import load_env
from common.llama_settings import configure_llama
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    download_loader,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load an existing index from disk or create a new one.")
    parser.add_argument("--persist_dir", type=str, default="./storage", help="Directory for stored index data.")
    parser.add_argument(
        "--pages",
        nargs="+",
        default=["Natural Language Processing", "Artificial Intelligence"],
        help="List of Wikipedia page titles to load if a new index must be created.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What does NLP stand for?",
        help="Question to ask after the index is loaded or created.",
    )
    return parser.parse_args()


def main() -> None:
    load_env()
    configure_llama()
    args = parse_args()

    if not os.path.exists(args.persist_dir):
        print("Storage directory not found. Creating a new index...")

        WikipediaReader = download_loader("WikipediaReader")
        loader = WikipediaReader()
        documents = loader.load_data(pages=args.pages)

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[splitter],
        )
        index.storage_context.persist(persist_dir=args.persist_dir)
    else:
        print("Storage directory found. Loading existing index...")

        storage_context = StorageContext.from_defaults(persist_dir=args.persist_dir)
        index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(args.query)

    print("\nQuestion:")
    print(args.query)

    print("\nAnswer:")
    print(str(response))

    print("\n--- SOURCES ---")
    for i, node in enumerate(response.source_nodes, 1):
        print(f"\nSource {i} (score: {node.score}):")
        print(node.node.text[:300])


if __name__ == "__main__":
    main()