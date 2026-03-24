"""
Example 3: Low-level retriever demo with similarity_top_k.

This script indexes a GitHub repository and retrieves source nodes using
VectorIndexRetriever. It helps inspect how top-k affects retrieval.

Usage:
    python m_04_003_retriever_topk_demo.py --github_url https://github.com/owner/repo --question "What is this repository about?" --similarity_top_k 4
"""

import argparse
import os
import re
import sys
from pathlib import Path

import nest_asyncio

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.env_loader import load_env
from common.llama_settings import configure_llama

from llama_index.readers.github import GithubClient, GithubRepositoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect retrieval on a GitHub repository index.")
    parser.add_argument("--github_url", type=str, required=True, help="GitHub repository URL.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask.")
    parser.add_argument("--branch", type=str, default="main", help="Git branch.")
    parser.add_argument("--similarity_top_k", type=int, default=4, help="Top-k nodes to retrieve.")
    return parser.parse_args()


def parse_github_url(url: str) -> tuple[str | None, str | None]:
    pattern = r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$"
    match = re.match(pattern, url.strip())
    return match.groups() if match else (None, None)


def main() -> None:
    nest_asyncio.apply()
    load_env()
    configure_llama()
    args = parse_args()

    owner, repo = parse_github_url(args.github_url)
    if not owner or not repo:
        raise ValueError("Invalid GitHub URL.")

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise EnvironmentError("GITHUB_TOKEN not found in environment variables.")

    github_client = GithubClient(github_token=github_token)
    loader = GithubRepositoryReader(
        github_client=github_client,
        owner=owner,
        repo=repo,
        filter_file_extensions=(
            [".py", ".js", ".ts", ".md"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        verbose=False,
        concurrent_requests=5,
    )

    docs = loader.load_data(branch=args.branch)
    index = VectorStoreIndex.from_documents(docs)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=args.similarity_top_k)
    nodes = retriever.retrieve(args.question)

    print("Question:")
    print(args.question)
    print(f"\nRetrieved nodes: {len(nodes)}")
    print(f"similarity_top_k={args.similarity_top_k}")

    for i, node in enumerate(nodes, start=1):
        print(f"\n--- Node {i} (score: {node.score}) ---")
        print(node.node.metadata)
        print(node.node.text[:600])


if __name__ == "__main__":
    main()
