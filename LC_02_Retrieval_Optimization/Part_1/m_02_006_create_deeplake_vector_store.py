"""
m_02_006_create_deeplake_vector_store.py

Creates a Deep Lake vector store dataset path for the Paul Graham essay example.

Usage:
    python m_02_006_create_deeplake_vector_store.py --org-id your-org-id
    python m_02_006_create_deeplake_vector_store.py --org-id genai360 --dataset-name LlamaIndex_paulgraham_essay
"""

from __future__ import annotations

import argparse

from m_02_003_common import create_deeplake_vector_store
from common.env_loader import load_env

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Deep Lake vector store.")
    parser.add_argument("--org-id", required=True, help="Activeloop organization ID.")
    parser.add_argument("--dataset-name", default="LlamaIndex_paulgraham_essay", help="Deep Lake dataset name.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing dataset if it exists.")
    return parser.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    _, dataset_path = create_deeplake_vector_store(
        org_id=args.org_id,
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
    )
    print(f"Vector store ready: {dataset_path}")


if __name__ == "__main__":
    main()
