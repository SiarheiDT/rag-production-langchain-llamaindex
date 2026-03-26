"""
m_03_006_shopping_agent_with_weather.py

Build a shopping agent that can take weather into account.

Tools:
1. inventory query tool
2. current date tool
3. total price calculation tool
4. mock weather tool

Example:
    python run.py --module 3 --part 3 --task 6 -- \
      --persist-dir LC_03_RAG_Agent_Systems/Part_3/storage/product_index \
      --question "Recommend a casual outfit for a woman for a rainy day under 80 dollars in Wroclaw"
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date
from pathlib import Path
from typing import Any

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool

from common.env_loader import load_env
from common.llama_settings import build_llama_settings
from common.common_output import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a weather-aware shopping agent.")
    parser.add_argument("--persist-dir", required=True, help="Persisted product index directory.")
    parser.add_argument("--question", required=True, help="Shopping request for the agent.")
    parser.add_argument("--top-k", type=int, default=3, help="Similarity top-k for inventory retrieval.")
    parser.add_argument("--verbose", action="store_true", help="Reserved flag for CLI consistency.")
    return parser.parse_args()


def validate_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Persist dir not found: {path_str}")
    if not path.is_dir():
        raise ValueError(f"Persist path is not a directory: {path_str}")
    return path


def get_current_date() -> str:
    """Return today's date in ISO format."""
    return date.today().isoformat()


def calculate_total_price(prices: list[float]) -> float:
    """Calculate the total price for a list of numeric prices."""
    return round(sum(prices), 2)


def get_weather(location: str) -> dict[str, Any]:
    """
    Mock weather tool.

    In production, this would call a real weather API.
    """
    weather_map = {
        "wroclaw": {"condition": "rainy", "temperature_c": 10},
        "warsaw": {"condition": "cloudy", "temperature_c": 12},
        "krakow": {"condition": "sunny", "temperature_c": 18},
        "london": {"condition": "rainy", "temperature_c": 9},
    }

    key = location.strip().lower()
    result = weather_map.get(key, {"condition": "mild", "temperature_c": 15})

    return {
        "location": location,
        "condition": result["condition"],
        "temperature_c": result["temperature_c"],
        "source": "mock_weather_tool",
    }


async def run_agent(persist_dir: str, question: str, top_k: int) -> str:
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(similarity_top_k=top_k)

    inventory_query_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="inventory_query_tool",
        description=(
            "Use this tool to search the shopping inventory for products "
            "based on user preferences, category, style, and price constraints."
        ),
    )

    current_date_tool = FunctionTool.from_defaults(
        fn=get_current_date,
        name="get_current_date",
        description="Use this tool to get today's date.",
    )

    total_price_tool = FunctionTool.from_defaults(
        fn=calculate_total_price,
        name="calculate_total_price",
        description="Use this tool to calculate the total price of selected products.",
    )

    weather_tool = FunctionTool.from_defaults(
        fn=get_weather,
        name="get_weather",
        description=(
            "Use this tool to get the weather for a city when weather conditions "
            "are relevant for outfit recommendations."
        ),
    )

    agent = FunctionAgent(
        tools=[
            inventory_query_tool,
            current_date_tool,
            total_price_tool,
            weather_tool,
        ],
        llm=Settings.llm,
        system_prompt=(
            "You are a shopping assistant. "
            "Help the user find suitable products from the inventory. "
            "Use the inventory tool for product search. "
            "Use the weather tool when the user's request mentions weather, rain, sun, cold, hot, "
            "or a location that may affect outfit choice. "
            "Use the total price tool when combining multiple product prices. "
            "Use the current date tool only when date awareness is useful. "
            "Keep answers practical, concise, and recommendation-focused."
        ),
    )

    response = await agent.run(question)
    return str(response)


def main() -> None:
    args = parse_args()

    load_env()
    build_llama_settings()

    persist_dir = validate_dir(args.persist_dir)

    answer = asyncio.run(
        run_agent(
            persist_dir=str(persist_dir),
            question=args.question,
            top_k=args.top_k,
        )
    )

    output_text = []
    output_text.append("=== SHOPPING AGENT WITH WEATHER RESULT ===")
    output_text.append(f"Persist dir: {persist_dir}")
    output_text.append(f"Question: {args.question}")
    output_text.append(f"Top-k: {args.top_k}")
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