"""
m_01_004_multi_document_agent.py

Build a multi-document FunctionAgent with two separate RAG tools,
one for Tesla and one for Apple, and let the agent decide which
document tool(s) to use.

Example:
    python run.py --module 3 --part 1 --task 4 -- \
      --tesla-file LC_03_RAG_Agent_Systems/Part_1/data/tesla_earnings.txt \
      --apple-file LC_03_RAG_Agent_Systems/Part_1/data/apple_earnings.txt \
      --question "Compare Tesla and Apple growth themes." \
      --verbose
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool

from common.env_loader import load_env
from common.llama_settings import build_llama_settings
from common.common_output import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a multi-document FunctionAgent with Tesla and Apple RAG tools."
    )
    parser.add_argument(
        "--tesla-file",
        required=True,
        help="Path to the Tesla text file.",
    )
    parser.add_argument(
        "--apple-file",
        required=True,
        help="Path to the Apple text file.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question for the agent.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k retrieval value for each document query engine.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Reserved flag for CLI consistency.",
    )
    return parser.parse_args()


def validate_input_file(input_file: str) -> Path:
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {input_file}")
    return path


def build_query_tool(
    input_file: str,
    tool_name: str,
    description: str,
    top_k: int,
) -> QueryEngineTool:
    """
    Build a QueryEngineTool for a single source document.
    """
    documents = SimpleDirectoryReader(input_files=[input_file]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    return QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=tool_name,
        description=description,
    )


async def run_agent(
    tesla_file: str,
    apple_file: str,
    question: str,
    top_k: int,
) -> str:
    tesla_tool = build_query_tool(
        input_file=tesla_file,
        tool_name="tesla_financials",
        description=(
            "Use this tool for questions about Tesla, including revenue, growth themes, "
            "production scaling, autonomous driving, energy products, risks, and forecasts."
        ),
        top_k=top_k,
    )

    apple_tool = build_query_tool(
        input_file=apple_file,
        tool_name="apple_financials",
        description=(
            "Use this tool for questions about Apple, including revenue, growth themes, "
            "services growth, hardware business, ecosystem strategy, risks, and forecasts."
        ),
        top_k=top_k,
    )

    agent = FunctionAgent(
        tools=[tesla_tool, apple_tool],
        llm=Settings.llm,
        system_prompt=(
            "You are a financial document comparison assistant. "
            "Use the Tesla tool only for Tesla-specific questions, "
            "the Apple tool only for Apple-specific questions, "
            "and use both tools when the user asks for comparison, contrast, or cross-company analysis. "
            "Ground the answer in the available documents."
        ),
    )

    response = await agent.run(question)
    return str(response)


def main() -> None:
    args = parse_args()

    load_env()
    build_llama_settings()

    tesla_path = validate_input_file(args.tesla_file)
    apple_path = validate_input_file(args.apple_file)

    answer = asyncio.run(
        run_agent(
            tesla_file=str(tesla_path),
            apple_file=str(apple_path),
            question=args.question,
            top_k=args.top_k,
        )
    )

    output_text = []
    output_text.append("=== MULTI-DOCUMENT AGENT RESULT ===")
    output_text.append(f"Tesla file: {tesla_path}")
    output_text.append(f"Apple file: {apple_path}")
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
