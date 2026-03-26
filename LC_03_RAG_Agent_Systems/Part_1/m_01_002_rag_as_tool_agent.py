"""
m_01_002_rag_as_tool_agent.py

Wrap a RAG query engine as a tool and expose it to a modern LlamaIndex FunctionAgent.

Example:
    python run.py --module 3 --part 1 --task 2 -- \
      --input-file LC_03_RAG_Agent_Systems/Part_1/data/tesla_earnings.txt \
      --question "Summarize the forward-looking statements in this document." \
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
    parser = argparse.ArgumentParser(description="Create a FunctionAgent over a RAG tool.")
    parser.add_argument("--input-file", required=True, help="Path to the input text file.")
    parser.add_argument("--question", required=True, help="Question for the agent.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k retrieval value.")
    parser.add_argument("--verbose", action="store_true", help="Reserved flag for CLI consistency.")
    return parser.parse_args()


def validate_input_file(input_file: str) -> Path:
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {input_file}")
    return path


async def run_agent(input_file: str, question: str, top_k: int) -> str:
    documents = SimpleDirectoryReader(input_files=[input_file]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    rag_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="financial_document_rag",
        description=(
            "Use this tool to answer questions about the provided financial document. "
            "It is useful for summaries, risks, forecasts, metrics, and forward-looking statements."
        ),
    )

    agent = FunctionAgent(
        tools=[rag_tool],
        llm=Settings.llm,
        system_prompt=(
            "You are a helpful financial document assistant. "
            "Use the retrieval tool whenever the question depends on the source document."
        ),
    )

    response = await agent.run(question)
    return str(response)


def main() -> None:
    args = parse_args()
    load_env()
    build_llama_settings()

    input_path = validate_input_file(args.input_file)
    answer = asyncio.run(run_agent(str(input_path), args.question, args.top_k))

    output_text = []
    output_text.append("=== RAG AS TOOL AGENT RESULT ===")
    output_text.append(f"Input file: {input_path}")
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