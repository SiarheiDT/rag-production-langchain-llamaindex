"""
m_03_002_build_product_rag_index.py

Build a product RAG index from a cleaned shopping catalog.

Example:
    python run.py --module 3 --part 3 --task 2 -- \
      --input-file LC_03_RAG_Agent_Systems/Part_3/data/shopping_catalog.csv \
      --persist-dir LC_03_RAG_Agent_Systems/Part_3/storage/product_index
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core import load_index_from_storage

from common.env_loader import load_env
from common.llama_settings import build_llama_settings
from common.common_output import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a product RAG index.")
    parser.add_argument("--input-file", required=True, help="Path to the cleaned catalog CSV.")
    parser.add_argument("--persist-dir", required=True, help="Directory to persist the index.")
    return parser.parse_args()


def validate_input_file(input_file: str) -> Path:
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {input_file}")
    return path


def build_product_text(row: pd.Series) -> str:
    return f"""
# Product Name
{row["name"]}

# Category
{row["category"]}

# Gender
{row["gender"]}

# Price
{row["price"]}

# Product ID
{row["product_id"]}

# Description
{row["description"]}
""".strip()


def main() -> None:
    args = parse_args()

    load_env()
    build_llama_settings()

    input_path = validate_input_file(args.input_file)
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    documents = []
    for _, row in df.iterrows():
        doc = Document(
            text=build_product_text(row),
            metadata={
                "product_id": str(row["product_id"]),
                "name": str(row["name"]),
                "category": str(row["category"]),
                "gender": str(row["gender"]),
                "price": float(row["price"]),
            },
        )
        documents.append(doc)

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(persist_dir))

    output_text = []
    output_text.append("=== PRODUCT RAG INDEX BUILD RESULT ===")
    output_text.append(f"Input file: {input_path}")
    output_text.append(f"Persist dir: {persist_dir}")
    output_text.append(f"Documents indexed: {len(documents)}")

    result = "\n".join(output_text)
    saved_path = save_result(__file__, result)

    print(result)
    print()
    print(f"Saved to: {saved_path}")


if __name__ == "__main__":
    main()