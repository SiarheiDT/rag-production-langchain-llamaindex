"""
Example 6: LCEL pipeline with PromptTemplate, ChatOpenAI and StrOutputParser.

This script reproduces the lesson example that builds a simple runnable
using LangChain Expression Language (LCEL).

Usage:
    export OPENAI_API_KEY=...
    python m_01_006_lcel_rainbow.py
"""

from common.env_loader import load_env
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main() -> None:
    load_env()

    prompt = PromptTemplate.from_template(
        "List all the colors in a {item}."
    )

    runnable = prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0) | StrOutputParser()
    result = runnable.invoke({"item": "rainbow"})

    print(result)


if __name__ == "__main__":
    main()
