"""
m_02_005_create_nodes.py

Splits the Paul Graham essay documents into nodes/chunks using the same chunking
pattern shown in the module examples.

Usage:
    python m_02_005_create_nodes.py
    python m_02_005_create_nodes.py --chunk-size 512 --chunk-overlap 64
"""

from __future__ import annotations

import argparse

from m_02_003_common import build_nodes, load_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create nodes from documents.")
    parser.add_argument("--data-dir", default="./data/paul_graham", help="Directory containing source files.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size for node parsing.")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap for node parsing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(data_dir=args.data_dir)
    nodes = build_nodes(
        documents=documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Created nodes: {len(nodes)}")
    if nodes:
        print("\nFirst node preview:\n")
        print(nodes[0].get_content()[:1000])


if __name__ == "__main__":
    main()
