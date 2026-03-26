"""
m_02_002_basic_langsmith_trace.py

Purpose:
    Create a minimal LangChain + LangSmith traced run.

What it demonstrates:
    - loading environment variables
    - creating a ChatOpenAI model
    - sending a small prompt
    - allowing LangSmith tracing to capture the run

Notes:
    This script assumes:
    - OPENAI_API_KEY is valid
    - LANGSMITH_API_KEY is valid
    - LANGSMITH_TRACING_V2=true

Run:
    python m_02_002_basic_langsmith_trace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from common.env_loader import load_env
    load_env()
except Exception:
    try:
        from dotenv import load_dotenv
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")
    except Exception:
        pass


def main() -> int:
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except Exception as exc:
        print("Missing dependency.")
        print(f"Details: {exc}")
        print("Install with: pip install langchain langchain-openai")
        return 1

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise assistant."),
            ("human", "Explain in 4-5 lines why tracing matters in LLM systems."),
        ]
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | model | StrOutputParser()

    print("=" * 88)
    print("Basic LangSmith Trace Demo")
    print("=" * 88)
    result = chain.invoke({})
    print(result)
    print("-" * 88)
    print("If LangSmith tracing is configured, this run should appear in your project dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
