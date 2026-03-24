"""
Example 3: Basic ChatOpenAI invocation with SystemMessage and HumanMessage.

This script mirrors the lesson example where the model answers:
"What is the capital of France?"

Usage:
    export OPENAI_API_KEY=...
    python m_01_003_chat_openai_basic.py
"""

from common.env_loader import load_env
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def main() -> None:
    load_env()

    chat = ChatOpenAI(model="gpt-3.5-turbo")

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
    ]

    response = chat.invoke(messages)

    print("Model response:")
    print(response)


if __name__ == "__main__":
    main()
