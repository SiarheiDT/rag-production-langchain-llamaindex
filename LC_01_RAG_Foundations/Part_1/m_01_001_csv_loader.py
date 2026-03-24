"""
Example 1: Load CSV data with LangChain CSVLoader.

This script demonstrates how to load CSV rows into LangChain Document objects.
It is based directly on the example shown in the lesson
"LangChain: Basic Concepts Recap".

Usage:
    python m_01_001_csv_loader.py --csv_path ./data/raw/data.csv
"""

import argparse
from common.env_loader import load_env
from langchain.document_loaders import CSVLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a CSV file into LangChain Document objects.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file.")
    return parser.parse_args()


def main() -> None:
    load_env()

    args = parse_args()

    loader = CSVLoader(args.csv_path)
    documents = loader.load()

    print(f"Loaded documents: {len(documents)}")
    for i, document in enumerate(documents[:5], start=1):
        print(f"\n--- Document {i} ---")
        print("Content:")
        print(document.page_content)
        print("Metadata:")
        print(document.metadata)


if __name__ == "__main__":
    main()
