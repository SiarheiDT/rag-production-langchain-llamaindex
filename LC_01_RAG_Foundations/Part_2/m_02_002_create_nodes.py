"""
Example 2: Convert documents into nodes with SimpleNodeParser.

This script follows the lecture example that uses WikipediaReader
and SimpleNodeParser to split documents into nodes.

Usage:
    python m_02_002_create_nodes.py --chunk_size 512 --chunk_overlap 20
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.env_loader import load_env
from llama_index.core import download_loader
from llama_index.core.node_parser import SimpleNodeParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LlamaIndex nodes from documents.")
    parser.add_argument("--chunk_size", type=int, default=512, help="Node chunk size.")
    parser.add_argument("--chunk_overlap", type=int, default=20, help="Node chunk overlap.")
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

    parser = SimpleNodeParser.from_defaults(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    nodes = parser.get_nodes_from_documents(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created nodes: {len(nodes)}")

    for i, node in enumerate(nodes[:5], start=1):
        print(f"\n--- Node {i} ---")
        print(node.text[:500])


if __name__ == "__main__":
    main()
