"""
Example 5: LCEL version of rainbow prompt (modern LangChain)

Replaces deprecated LLMChain with LCEL pipeline.

Usage:
    python m_01_005_llm_chain_rainbow.py
"""

import sys
from pathlib import Path

# add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.env_loader import load_env
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main() -> None:
    load_env()

    prompt = PromptTemplate.from_template(
        "List all the colors in a rainbow"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({})

    print(result)


if __name__ == "__main__":
    main()