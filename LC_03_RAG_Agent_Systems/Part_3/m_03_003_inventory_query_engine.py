"""
m_03_003_inventory_query_engine.py

Query a persisted product RAG index.

Example:
    python run.py --module 3 --part 3 --task 3 -- \
      --persist-dir LC_03_RAG_Agent_Systems/Part_3/storage/product_index \
      --question "Find a casual women blouse under 30 dollars"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage

from common.env_loader import load_env
from common.llama_settings import build_llama_settings
from common.common_output import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a product RAG index.")
    parser.add_argument("--persist-dir", required=True, help="Persisted index directory.")
    parser.add_argument("--question", required=True, help="User query.")
    parser.add_argument("--top-k", type=int, default=3, help="Similarity top-k.")
    return parser.parse_args()


def validate_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Persist dir not found: {path_str}")
    if not path.is_dir():
        raise ValueError(f"Persist path is not a directory: {path_str}")
    return path


def main() -> None:
    args = parse_args()

    load_env()
    build_llama_settings()

    persist_dir = validate_dir(args.persist_dir)

    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(similarity_top_k=args.top_k)
    response = query_engine.query(args.question)

    output_text = []
    output_text.append("=== INVENTORY QUERY ENGINE RESULT ===")
    output_text.append(f"Persist dir: {persist_dir}")
    output_text.append(f"Question: {args.question}")
    output_text.append(f"Top-k: {args.top_k}")
    output_text.append("")
    output_text.append("=== ANSWER ===")
    output_text.append(str(response))

    result = "\n".join(output_text)
    saved_path = save_result(__file__, result)

    print(result)
    print()
    print(f"Saved to: {saved_path}")


if __name__ == "__main__":
    main()