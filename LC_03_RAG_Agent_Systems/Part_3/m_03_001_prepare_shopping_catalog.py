"""
m_03_001_prepare_shopping_catalog.py

Prepare a clean shopping catalog from a raw CSV file.

Example:
    python run.py --module 3 --part 3 --task 1 -- \
      --input-file LC_03_RAG_Agent_Systems/Part_3/data/shopping_catalog_raw.csv \
      --output-file LC_03_RAG_Agent_Systems/Part_3/data/shopping_catalog.csv

Notes:
- This script keeps the catalog simple and interview-friendly.
- It creates a normalized dataset for the downstream RAG pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.common_output import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a clean shopping catalog.")
    parser.add_argument("--input-file", required=True, help="Path to the raw input CSV.")
    parser.add_argument("--output-file", required=True, help="Path to the cleaned output CSV.")
    return parser.parse_args()


def validate_input_file(input_file: str) -> Path:
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {input_file}")
    return path


def infer_gender(category: str) -> str:
    category_lower = str(category).lower()

    if "women" in category_lower:
        return "women"
    if "men" in category_lower:
        return "men"
    return "either"


def main() -> None:
    args = parse_args()

    input_path = validate_input_file(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_path)

    required_columns = ["product_id", "name", "category", "price", "description"]
    missing = [col for col in required_columns if col not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    df = pd.DataFrame(
        {
            "product_id": df_raw["product_id"].astype(str),
            "name": df_raw["name"].fillna("").astype(str),
            "category": df_raw["category"].fillna("").astype(str),
            "price": pd.to_numeric(df_raw["price"], errors="coerce"),
            "description": df_raw["description"].fillna("").astype(str),
        }
    )

    df = df.dropna(subset=["price"]).copy()
    df["gender"] = df["category"].apply(infer_gender)

    df = df[
        ["product_id", "name", "category", "gender", "price", "description"]
    ].reset_index(drop=True)

    df.to_csv(output_path, index=False)

    output_text = []
    output_text.append("=== SHOPPING CATALOG PREPARATION RESULT ===")
    output_text.append(f"Input file: {input_path}")
    output_text.append(f"Output file: {output_path}")
    output_text.append(f"Rows written: {len(df)}")
    output_text.append("")
    output_text.append("=== SAMPLE ROWS ===")
    output_text.append(df.head(5).to_string(index=False))

    result = "\n".join(output_text)
    saved_path = save_result(__file__, result)

    print(result)
    print()
    print(f"Saved to: {saved_path}")


if __name__ == "__main__":
    main()