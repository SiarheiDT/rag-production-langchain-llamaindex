"""
Example 2: Load Wikipedia content with WikipediaLoader.

This script demonstrates how to fetch a Wikipedia article and convert it
into LangChain Document objects.

Usage:
    python m_01_002_wikipedia_loader.py --query "Machine_learning"
"""

import argparse
from common.env_loader import load_env
from langchain_community.document_loaders import WikipediaLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Wikipedia content into LangChain Document objects.")
    parser.add_argument("--query", type=str, default="Machine_learning", help="Wikipedia query/title.")
    return parser.parse_args()


def main() -> None:
    load_env()

    args = parse_args()

    loader = WikipediaLoader(args.query)
    documents = loader.load()

    print(f"Loaded documents: {len(documents)}")
    for i, document in enumerate(documents[:3], start=1):
        print(f"\n--- Document {i} ---")
        print("Content preview:")
        print(document.page_content[:1000])
        print("\nMetadata:")
        print(document.metadata)


if __name__ == "__main__":
    main()
