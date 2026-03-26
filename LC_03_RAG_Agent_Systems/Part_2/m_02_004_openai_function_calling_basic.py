"""
m_02_004_openai_function_calling_basic.py

Demonstrate basic function calling with the modern OpenAI Responses API.

This script:
1. Sends a user question to the model with custom function tools.
2. Checks whether the model asked to call a function.
3. Executes the local Python function.
4. Sends the function result back to the model.
5. Prints the final answer.

Example:
    python run.py --module 3 --part 2 --task 4 -- \
      --question "What is the weather in Wroclaw in celsius?"

    python run.py --module 3 --part 2 --task 4 -- \
      --question "Multiply 12 by 7"
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from openai import OpenAI

from common.env_loader import load_env
from common.common_output import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenAI Responses API with basic function calling."
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
        "--tool-choice",
        default="auto",
        choices=["auto", "required", "none"],
        help="How the model should choose tools.",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a helpful assistant. "
            "Use tools when they are useful, and explain results clearly."
        ),
        help="System prompt for the assistant.",
    )
    return parser.parse_args()


def get_current_weather(location: str, unit: str) -> dict[str, Any]:
    """
    Mock weather function.

    In a production system, this would call a real external weather API.
    """
    fake_weather = {
        "wroclaw": {"celsius": 16, "fahrenheit": 61},
        "warsaw": {"celsius": 15, "fahrenheit": 59},
        "krakow": {"celsius": 14, "fahrenheit": 57},
        "london": {"celsius": 12, "fahrenheit": 54},
    }

    key = location.strip().lower()
    temp_info = fake_weather.get(key, {"celsius": 20, "fahrenheit": 68})

    return {
        "location": location,
        "unit": unit,
        "temperature": temp_info[unit],
        "conditions": "partly cloudy",
        "source": "mock_weather_service",
    }


def multiply_numbers(a: float, b: float) -> dict[str, Any]:
    """
    Simple multiplication function.
    """
    return {
        "a": a,
        "b": b,
        "result": a * b,
    }


def get_tools() -> list[dict[str, Any]]:
    """
    Tool definitions for the OpenAI Responses API.
    """
    return [
        {
            "type": "function",
            "name": "get_current_weather",
            "description": "Get the current weather in a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, for example Wroclaw or London.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit.",
                    },
                },
                "required": ["location", "unit"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "multiply_numbers",
            "description": "Multiply two numeric values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number.",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number.",
                    },
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Dispatch tool execution to a local Python function.
    """
    if tool_name == "get_current_weather":
        return get_current_weather(
            location=arguments["location"],
            unit=arguments["unit"],
        )

    if tool_name == "multiply_numbers":
        return multiply_numbers(
            a=arguments["a"],
            b=arguments["b"],
        )

    raise ValueError(f"Unknown tool requested: {tool_name}")


def extract_function_calls(response: Any) -> list[Any]:
    """
    Extract function call items from a Responses API response.
    """
    calls = []

    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "function_call":
            calls.append(item)

    return calls


def main() -> None:
    args = parse_args()

    load_env()
    client = OpenAI()

    tools = get_tools()

    first_response = client.responses.create(
        model=args.model,
        instructions=args.system_prompt,
        input=args.question,
        tools=tools,
        tool_choice=args.tool_choice,
    )

    function_calls = extract_function_calls(first_response)

    executed_tools_log: list[str] = []

    # If no function call was requested, use the first model answer directly.
    if not function_calls:
        final_answer = first_response.output_text
    else:
        followup_items: list[dict[str, Any]] = []

        for call in function_calls:
            tool_name = call.name
            arguments = json.loads(call.arguments)
            tool_result = execute_tool(tool_name, arguments)

            executed_tools_log.append(
                f"{tool_name}({json.dumps(arguments, ensure_ascii=False)}) -> "
                f"{json.dumps(tool_result, ensure_ascii=False)}"
            )

            followup_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(tool_result, ensure_ascii=False),
                }
            )

        second_response = client.responses.create(
            model=args.model,
            previous_response_id=first_response.id,
            input=followup_items,
            tools=tools,
            tool_choice="none",
        )
        final_answer = second_response.output_text

    output_text = []
    output_text.append("=== OPENAI FUNCTION CALLING BASIC RESULT ===")
    output_text.append(f"Model: {args.model}")
    output_text.append(f"Tool choice: {args.tool_choice}")
    output_text.append(f"Question: {args.question}")
    output_text.append("")

    if executed_tools_log:
        output_text.append("=== EXECUTED TOOLS ===")
        output_text.extend(executed_tools_log)
        output_text.append("")
    else:
        output_text.append("=== EXECUTED TOOLS ===")
        output_text.append("No tool was called.")
        output_text.append("")

    output_text.append("=== ANSWER ===")
    output_text.append(final_answer)

    result = "\n".join(output_text)
    saved_path = save_result(__file__, result)

    print(result)
    print()
    print(f"Saved to: {saved_path}")


if __name__ == "__main__":
    main()