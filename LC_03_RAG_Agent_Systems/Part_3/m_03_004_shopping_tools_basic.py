"""
m_03_004_shopping_tools_basic.py

Build basic shopping tools:
1. inventory query tool over a persisted product index
2. current date tool
3. total price calculation tool

Example:
    python run.py --module 3 --part 3 --task 4 -- \
      --persist-dir LC_03_RAG_Agent_Systems/Part_3/storage/product_index \
      --question "Find a casual women blouse under 30 dollars"

Notes:
- This script demonstrates how to wrap a persisted RAG query engine as a tool.
- It also provides simple utility tools used later by the shopping agent.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.tools import FunctionTool, QueryEngineTool

from common.env_loader import load_env
from common.llama_settings import build_llama_settings
from common.common_output import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and test basic shopping tools.")
    parser.add_argument("--persist-dir", required=True, help="Persisted product index directory.")
    parser.add_argument("--question", required=True, help="Question for the inventory query tool.")
    parser.add_argument("--top-k", type=int, default=3, help="Similarity top-k.")
    return parser.parse_args()


def validate_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Persist dir not found: {path_str}")
    if not path.is_dir():
        raise ValueError(f"Persist path is not a directory: {path_str}")
    return path


def get_current_date() -> str:
    """
    Return today's date in ISO format.
    """
    return date.today().isoformat()


def calculate_total_price(prices: list[float]) -> float:
    """
    Calculate the total price for a list of numeric prices.
    """
    return round(sum(prices), 2)


def main() -> None:
    args = parse_args()

    load_env()
    build_llama_settings()

    persist_dir = validate_dir(args.persist_dir)

    # Load the persisted product index.
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)

    # Create a standard query engine first.
    query_engine = index.as_query_engine(similarity_top_k=args.top_k)

    # Wrap the query engine as an inventory tool.
    inventory_query_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="inventory_query_tool",
        description=(
            "Use this tool to search the shopping inventory for products "
            "based on user preferences, category, style, and price constraints."
        ),
    )

    # Wrap utility functions as tools.
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

    # Test the inventory query tool directly.
    inventory_response = inventory_query_tool.query_engine.query(args.question)

    # Test utility tools directly as plain functions.
    today_value = get_current_date()
    sample_total = calculate_total_price([24.99, 29.99])

    output_text = []
    output_text.append("=== SHOPPING TOOLS BASIC RESULT ===")
    output_text.append(f"Persist dir: {persist_dir}")
    output_text.append(f"Question: {args.question}")
    output_text.append(f"Top-k: {args.top_k}")
    output_text.append("")
    output_text.append("=== REGISTERED TOOLS ===")
    output_text.append(f"- {inventory_query_tool.metadata.name}")
    output_text.append(f"- {current_date_tool.metadata.name}")
    output_text.append(f"- {total_price_tool.metadata.name}")
    output_text.append("")
    output_text.append("=== INVENTORY TOOL ANSWER ===")
    output_text.append(str(inventory_response))
    output_text.append("")
    output_text.append("=== UTILITY TOOL TESTS ===")
    output_text.append(f"Current date: {today_value}")
    output_text.append(f"Sample total price [24.99, 29.99]: {sample_total}")

    result = "\n".join(output_text)
    saved_path = save_result(__file__, result)

    print(result)
    print()
    print(f"Saved to: {saved_path}")


if __name__ == "__main__":
    main()