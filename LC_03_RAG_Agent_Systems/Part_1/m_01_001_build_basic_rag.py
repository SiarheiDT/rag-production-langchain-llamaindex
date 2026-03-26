"""
m_01_001_build_basic_rag.py

Builds a basic RAG pipeline with LlamaIndex from a local text file
and runs a single query against the created index.

Example:
    python run.py --module 3 --part 1 --task 1 -- \
      --input-file LC_03_RAG_Agent_Systems/Part_1/data/tesla_earnings.txt \
      --question "What does the document say about future growth?"

Notes:
- This script uses a local file as the knowledge source.
- The result is saved with common_output.save_result().
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from common.env_loader import load_env
from common.llama_settings import build_llama_settings
from common.common_output import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a basic RAG index and query it.")
    parser.add_argument("--input-file", required=True, help="Path to the input text file.")
    parser.add_argument("--question", required=True, help="Question to ask the RAG system.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k similarity results.")
    return parser.parse_args()


def validate_input_file(input_file: str) -> Path:
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {input_file}")
    return path


def main() -> None:
    args = parse_args()

    # Load environment variables such as OPENAI_API_KEY.
    load_env()

    # Apply shared LlamaIndex settings from the project helper.
    build_llama_settings()

    input_path = validate_input_file(args.input_file)

    # Read the document into LlamaIndex document objects.
    documents = SimpleDirectoryReader(input_files=[str(input_path)]).load_data()

    # Build an in-memory vector index from documents.
    index = VectorStoreIndex.from_documents(documents)

    # Create a query engine over the index.
    query_engine = index.as_query_engine(similarity_top_k=args.top_k)

    # Run the user question through the RAG pipeline.
    response = query_engine.query(args.question)

    output_text = []
    output_text.append("=== BASIC RAG RESULT ===")
    output_text.append(f"Input file: {input_path}")
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