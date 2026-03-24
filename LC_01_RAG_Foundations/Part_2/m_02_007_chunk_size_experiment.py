"""
Task 7: Compare how chunk_size changes node creation.

This script loads the same documents and compares how many nodes
are created for different chunk sizes.

Usage:
    python m_02_007_chunk_size_experiment.py --chunk_sizes 256 512 1024
    python run.py --module 1 --part 2 --task 7 -- --chunk_sizes 256 512 1024
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.env_loader import load_env
from common.llama_settings import configure_llama
from llama_index.core import download_loader
from llama_index.core.node_parser import SentenceSplitter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare node counts for different chunk sizes."
    )
    parser.add_argument(
        "--chunk_sizes",
        nargs="+",
        type=int,
        default=[256, 512, 1024],
        help="Chunk sizes to compare.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="Overlap between chunks.",
    )
    parser.add_argument(
        "--pages",
        nargs="+",
        default=["Natural Language Processing", "Artificial Intelligence"],
        help="Wikipedia pages to load.",
    )
    parser.add_argument(
        "--preview_nodes",
        type=int,
        default=2,
        help="How many nodes to preview for each chunk size.",
    )
    return parser.parse_args()


def main() -> None:
    load_env()
    configure_llama()
    args = parse_args()

    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    documents = loader.load_data(pages=args.pages)

    print(f"Loaded documents: {len(documents)}")

    for chunk_size in args.chunk_sizes:
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents(documents)

        print("\n" + "=" * 80)
        print(f"chunk_size={chunk_size}, chunk_overlap={args.chunk_overlap}")
        print(f"created_nodes={len(nodes)}")

        for i, node in enumerate(nodes[:args.preview_nodes], start=1):
            print(f"\n--- Preview Node {i} ---")
            print(node.text[:300])


if __name__ == "__main__":
    main()