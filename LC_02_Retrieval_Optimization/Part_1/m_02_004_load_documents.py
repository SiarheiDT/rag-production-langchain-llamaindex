"""
m_02_004_load_documents.py

Loads the Paul Graham essay documents using SimpleDirectoryReader.

Usage:
    python m_02_004_load_documents.py
    python m_02_004_load_documents.py --data-dir ./data/paul_graham
"""

from __future__ import annotations

import argparse

from m_02_003_common import load_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load documents with SimpleDirectoryReader.")
    parser.add_argument("--data-dir", default="./data/paul_graham", help="Directory containing source text files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(data_dir=args.data_dir)

    print(f"Loaded documents: {len(documents)}")
    if documents:
        first = documents[0]
        text = first.text if hasattr(first, "text") else str(first)
        print("\nFirst 1000 characters:\n")
        print(text[:1000])


if __name__ == "__main__":
    main()
