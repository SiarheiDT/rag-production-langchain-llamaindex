"""
m_02_003_openai_responses_with_code_interpreter.py

Create an assistant-style interaction using the OpenAI Responses API
with the code interpreter tool enabled.

Example:
    python run.py --module 3 --part 2 --task 3 -- \
      --question "Solve the equation 3x + 11 = 14 and explain the steps."

Notes:
- This script uses the modern OpenAI Responses API.
- Code Interpreter requires a container configuration.
"""

from __future__ import annotations

import argparse

from openai import OpenAI

from common.env_loader import load_env
from common.common_output import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenAI Responses API with code interpreter."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask the assistant.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a precise technical assistant. "
            "Use code interpreter whenever computation or verification is useful."
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
        tools=[
            {
                "type": "code_interpreter",
                "container": {
                    "type": "auto",
                },
            }
        ],
        include=["code_interpreter_call.outputs"],
    )

    answer = response.output_text

    output_text = []
    output_text.append("=== OPENAI RESPONSES CODE INTERPRETER RESULT ===")
    output_text.append(f"Model: {args.model}")
    output_text.append("Container type: auto")
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