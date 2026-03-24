"""
Example 3: Create a VectorStoreIndex and query it.

This script follows the lecture example:
- load Wikipedia pages
- build a vector index
- create a query engine
- ask a question

Usage:
    python m_02_003_vector_index_query.py --query "What does NLP stand for?"
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.env_loader import load_env
from llama_index.core import VectorStoreIndex, download_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a VectorStoreIndex and query it.")
    parser.add_argument(
        "--query",
        type=str,
        default="What does NLP stand for?",
        help="Question to ask the query engine.",
    )
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

    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    documents = loader.load_data(pages=args.pages)

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(args.query)

    print("Question:")
    print(args.query)
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
