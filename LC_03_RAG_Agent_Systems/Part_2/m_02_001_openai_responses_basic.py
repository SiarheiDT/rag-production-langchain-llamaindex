"""
m_02_001_openai_responses_basic.py

Create a basic AI assistant interaction using the modern OpenAI Responses API.

Example:
    python run.py --module 3 --part 2 --task 1 -- \
      --question "Explain what an AI assistant is in simple terms."

Notes:
- This script uses the current OpenAI Responses API instead of the legacy Assistants API.
- The result is saved via common_output.save_result().
"""

from __future__ import annotations

import argparse

from openai import OpenAI

from common.env_loader import load_env
from common.common_output import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a basic assistant-style interaction via the OpenAI Responses API."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="User question for the assistant.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a helpful AI assistant. "
            "Provide clear, accurate, and concise answers."
        ),
        help="System prompt for the assistant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_env()
    client = OpenAI()

    response = client.responses.create(
        model=args.model,
        instructions=args.system_prompt,
        input=args.question,
    )

    answer = response.output_text

    output_text = []
    output_text.append("=== OPENAI RESPONSES BASIC RESULT ===")
    output_text.append(f"Model: {args.model}")
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