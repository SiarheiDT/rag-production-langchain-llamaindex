"""
Example 1: Load Wikipedia pages with LlamaIndex.

This script follows the lecture example that uses WikipediaReader
to load multiple pages into LlamaIndex Document objects.

Usage:
    python m_02_001_wikipedia_reader.py --pages "Natural Language Processing" "Artificial Intelligence"
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.env_loader import load_env
from llama_index.core import download_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Wikipedia pages with LlamaIndex.")
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

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()

    documents = loader.load_data(pages=args.pages)

    print(f"Loaded documents: {len(documents)}")
    for i, doc in enumerate(documents, start=1):
        print(f"\n--- Document {i} ---")
        text = getattr(doc, "text", str(doc))
        print(text[:1000])


if __name__ == "__main__":
    main()
