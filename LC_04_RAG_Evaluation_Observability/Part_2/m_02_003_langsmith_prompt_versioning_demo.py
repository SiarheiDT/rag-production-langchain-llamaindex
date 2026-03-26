"""
m_02_003_langsmith_prompt_versioning_demo.py

Purpose:
    Demonstrate the practical idea of prompt versioning for LangSmith / LangChain Hub.

What it does:
    - creates two local prompt versions
    - runs both against the same input
    - prints the outputs side by side

Why this matters:
    In production, prompt changes should be compared systematically.
    This file does not push to Hub automatically because that may not be
    desirable in every environment. Instead, it demonstrates the comparison
    pattern you should use before versioning/promoting a prompt.

Run:
    python m_02_003_langsmith_prompt_versioning_demo.py
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
        return 1

    topic = "RAG evaluation"

    prompt_v1 = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise technical assistant."),
            ("human", "Tell me a short joke about {topic}."),
        ]
    )

    prompt_v2 = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise technical assistant."),
            ("human", "Tell me a short but clearer joke about {topic}. Avoid generic wording."),
        ]
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain_v1 = prompt_v1 | model | parser
    chain_v2 = prompt_v2 | model | parser

    print("=" * 88)
    print("Prompt Versioning Demo")
    print("=" * 88)

    out_v1 = chain_v1.invoke({"topic": topic})
    out_v2 = chain_v2.invoke({"topic": topic})

    print("PROMPT V1 OUTPUT")
    print("-" * 88)
    print(out_v1)
    print("=" * 88)
    print("PROMPT V2 OUTPUT")
    print("-" * 88)
    print(out_v2)
    print("=" * 88)
    print("Use this side-by-side pattern before publishing or pinning a prompt version.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
