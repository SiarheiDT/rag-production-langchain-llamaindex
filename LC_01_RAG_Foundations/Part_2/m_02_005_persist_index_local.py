"""
Example 5: Persist a local LlamaIndex index to disk.

This script follows the lecture example that stores an index
locally using the storage_context.persist() method.

Usage:
    python m_02_005_persist_index_local.py --persist_dir ./storage
"""

import argparse
import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.env_loader import load_env
from common.llama_settings import configure_llama

from llama_index.core import VectorStoreIndex, download_loader
from llama_index.core.node_parser import SentenceSplitter

# Apply global LlamaIndex settings (embeddings, etc.)
configure_llama()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist a LlamaIndex index locally.")
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="./storage",
        help="Directory where the index will be saved."
    )
    parser.add_argument(
        "--pages",
        nargs="+",
        default=["Natural Language Processing", "Artificial Intelligence"],
        help="List of Wikipedia page titles to load.",
    )
    return parser.parse_args()


def main() -> None:
    load_env()              # ← сначала загрузили env
    configure_llama()       # ← потом инициализация LLM

    args = parse_args()

    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    documents = loader.load_data(pages=args.pages)

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter]
    )

    index.storage_context.persist(persist_dir=args.persist_dir)

    print("Index persisted successfully.")
    print(f"Persist dir: {args.persist_dir}")


if __name__ == "__main__":
    main()