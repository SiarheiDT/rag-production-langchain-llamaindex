"""
Example 4: Generate text embeddings with OpenAIEmbeddings.

This script follows the lesson example and embeds a small list of texts,
then prints the number of embeddings and their dimensionality.

Usage:
    export OPENAI_API_KEY=...
    python m_01_004_openai_embeddings.py
"""

from common.env_loader import load_env
from langchain_openai import OpenAIEmbeddings


def main() -> None:
    load_env()

    embeddings_model = OpenAIEmbeddings()

    texts = [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!",
    ]

    embeddings = embeddings_model.embed_documents(texts)

    print("Number of documents embedded:", len(embeddings))
    print("Dimension of each embedding:", len(embeddings[0]))


if __name__ == "__main__":
    main()
