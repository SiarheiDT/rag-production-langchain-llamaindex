"""
m_02_007_build_vector_index.py

Builds a full VectorStoreIndex on top of the Paul Graham essay and uploads nodes
to Deep Lake.

Usage:
    python m_02_007_build_vector_index.py --org-id your-org-id
    python m_02_007_build_vector_index.py --org-id your-org-id --overwrite
"""

from __future__ import annotations

import argparse

from m_02_003_common import build_vector_index

from common.env_loader import load_env

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a VectorStoreIndex backed by Deep Lake.")
    parser.add_argument("--org-id", required=True, help="Activeloop organization ID.")
    parser.add_argument("--dataset-name", default="LlamaIndex_paulgraham_essay", help="Deep Lake dataset name.")
    parser.add_argument("--data-dir", default="./data/paul_graham", help="Directory containing source files.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size used for node creation.")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap used for node creation.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing Deep Lake dataset.")
    return parser.parse_args()


def main() -> None:

    load_env()

    args = parse_args()
    _, dataset_path, nodes = build_vector_index(
        org_id=args.org_id,
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        overwrite=args.overwrite,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Index built successfully.")
    print(f"Dataset path: {dataset_path}")
    print(f"Uploaded nodes: {len(nodes)}")


if __name__ == "__main__":
    main()
