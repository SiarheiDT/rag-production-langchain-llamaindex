"""
m_01_005_dynamic_tool_retriever_agent.py

Build a dynamic tool-retrieval pipeline:
1. Create document-specific RAG tools
2. Index those tools
3. Retrieve the most relevant tools for the user's question
4. Run a FunctionAgent over the retrieved tools

Example:
    python run.py --module 3 --part 1 --task 5 -- \
      --tesla-file LC_03_RAG_Agent_Systems/Part_1/data/tesla_earnings.txt \
      --apple-file LC_03_RAG_Agent_Systems/Part_1/data/apple_earnings.txt \
      --question "Which company is more focused on AI in its growth strategy?" \
      --tool-top-k 2
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import QueryEngineTool

from common.env_loader import load_env
from common.llama_settings import build_llama_settings
from common.common_output import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dynamic tool-retrieval agent over multiple document tools."
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
        "--doc-top-k",
        type=int,
        default=3,
        help="Top-k retrieval value inside each document query engine.",
    )
    parser.add_argument(
        "--tool-top-k",
        type=int,
        default=2,
        help="How many tools to retrieve for the question.",
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
    doc_top_k: int,
) -> QueryEngineTool:
    """
    Build a QueryEngineTool for a single document.
    """
    documents = SimpleDirectoryReader(input_files=[input_file]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=doc_top_k)

    return QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=tool_name,
        description=description,
    )


def build_tool_index(tools: List[QueryEngineTool]) -> ObjectIndex:
    """
    Build an index over tools so that relevant tools can be retrieved dynamically.
    """
    return ObjectIndex.from_objects(tools)

async def run_agent(
    tesla_file: str,
    apple_file: str,
    question: str,
    doc_top_k: int,
    tool_top_k: int,
) -> tuple[list[str], str]:
    tesla_tool = build_query_tool(
        input_file=tesla_file,
        tool_name="tesla_financials",
        description=(
            "Tool for Tesla financial and strategic information: growth themes, "
            "vehicle deliveries, production capacity, energy products, autonomy, AI, and risks."
        ),
        doc_top_k=doc_top_k,
    )

    apple_tool = build_query_tool(
        input_file=apple_file,
        tool_name="apple_financials",
        description=(
            "Tool for Apple financial and strategic information: growth themes, "
            "services, ecosystem expansion, silicon investment, AI integration, and risks."
        ),
        doc_top_k=doc_top_k,
    )

    all_tools = [tesla_tool, apple_tool]

    tool_index = build_tool_index(all_tools)
    tool_retriever = tool_index.as_retriever(similarity_top_k=tool_top_k)

    # Optional: retrieve once only for reporting/debugging
    selected_tools = tool_retriever.retrieve(question)
    selected_tool_names = [tool.metadata.name for tool in selected_tools]

    agent = FunctionAgent(
        tool_retriever=tool_retriever,
        llm=Settings.llm,
        system_prompt=(
            "You are a dynamic financial analysis assistant. "
            "Use the retrieved tools to answer the user's question. "
            "Ground your answer in the selected source tools."
        ),
    )

    response = await agent.run(question)
    return selected_tool_names, str(response)

    response = await agent.run(question)
    return selected_tool_names, str(response)


def main() -> None:
    args = parse_args()

    load_env()
    build_llama_settings()

    tesla_path = validate_input_file(args.tesla_file)
    apple_path = validate_input_file(args.apple_file)

    selected_tool_names, answer = asyncio.run(
        run_agent(
            tesla_file=str(tesla_path),
            apple_file=str(apple_path),
            question=args.question,
            doc_top_k=args.doc_top_k,
            tool_top_k=args.tool_top_k,
        )
    )

    output_text = []
    output_text.append("=== DYNAMIC TOOL RETRIEVER AGENT RESULT ===")
    output_text.append(f"Tesla file: {tesla_path}")
    output_text.append(f"Apple file: {apple_path}")
    output_text.append(f"Question: {args.question}")
    output_text.append(f"Document top-k: {args.doc_top_k}")
    output_text.append(f"Tool top-k: {args.tool_top_k}")
    output_text.append(f"Selected tools: {', '.join(selected_tool_names)}")
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
