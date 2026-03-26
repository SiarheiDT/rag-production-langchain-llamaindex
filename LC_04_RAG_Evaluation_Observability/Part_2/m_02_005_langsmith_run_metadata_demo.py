"""
m_02_005_langsmith_run_metadata_demo.py

Purpose:
    Show how to attach metadata / tags to a traced run so that analysis in LangSmith
    becomes easier.

What it demonstrates:
    - RunnableConfig
    - tags
    - metadata
    - project-friendly run labeling

Run:
    python m_02_005_langsmith_run_metadata_demo.py
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
        from langchain_core.runnables import RunnableConfig
    except Exception as exc:
        print("Missing dependency.")
        print(f"Details: {exc}")
        return 1

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise technical assistant."),
            ("human", "In 5 lines explain why prompt version pinning matters in production."),
        ]
    )

    chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()

    config = RunnableConfig(
        tags=["module4", "part2", "langsmith", "prompt-versioning"],
        metadata={
            "module": "LC_04",
            "part": "Part_2",
            "script": "m_02_005",
            "purpose": "run_metadata_demo",
        },
    )

    print("=" * 88)
    print("LangSmith Run Metadata Demo")
    print("=" * 88)
    result = chain.invoke({}, config=config)
    print(result)
    print("-" * 88)
    print("Look for tags/metadata in the LangSmith trace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
