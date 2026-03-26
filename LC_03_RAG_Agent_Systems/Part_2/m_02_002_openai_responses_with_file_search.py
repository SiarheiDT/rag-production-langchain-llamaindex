"""
m_02_002_openai_responses_with_file_search.py

Create an assistant-style interaction using the OpenAI Responses API
with file search over an uploaded document.

Example:
    python run.py --module 3 --part 2 --task 2 -- \
      --input-file LC_03_RAG_Agent_Systems/Part_1/data/tesla_earnings.txt \
      --question "What are the main growth themes in this document?"

Notes:
- This script uses the modern OpenAI Responses API.
- It uploads a local file into a vector store via the SDK helper
  that handles upload + polling in one step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openai import OpenAI

from common.env_loader import load_env
from common.common_output import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenAI Responses API with file search over a local document."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the local file to upload for retrieval.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about the uploaded file.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name.",
    )
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

    load_env()
    client = OpenAI()

    input_path = validate_input_file(args.input_file)

    # Create a vector store first.
    vector_store = client.vector_stores.create(
        name=f"vs_{input_path.stem}"
    )

    # Upload the file and wait until ingestion finishes.
    with open(input_path, "rb") as f:
        file_batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=[f],
        )

    if file_batch.status != "completed":
        raise RuntimeError(
            f"Vector store ingestion did not complete successfully. "
            f"Status: {file_batch.status}, file_counts: {file_batch.file_counts}"
        )

    response = client.responses.create(
        model=args.model,
        input=args.question,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store.id],
            }
        ],
    )

    answer = response.output_text

    output_text = []
    output_text.append("=== OPENAI RESPONSES FILE SEARCH RESULT ===")
    output_text.append(f"Model: {args.model}")
    output_text.append(f"Input file: {input_path}")
    output_text.append(f"Vector store id: {vector_store.id}")
    output_text.append(f"File batch status: {file_batch.status}")
    output_text.append(f"File counts: {file_batch.file_counts}")
    output_text.append(f"Question: {args.question}")
    output_text.append("")
    output_text.append("=== ANSWER ===")
    output_text.append(answer)

    result = "\n".join(output_text)
    saved_path = save_result(__file__, result)

    print(result)
    print()
    print(f"Saved to: {saved_path}")


if __name__ == "__main__":
    main()