"""
Example 4: Custom RetrieverQueryEngine with SimilarityPostprocessor.

This script follows the low-level API section of the lesson.

Usage:
    python m_04_004_custom_query_engine.py --github_url https://github.com/owner/repo --question "what code is in this repository?"
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
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a custom LlamaIndex query engine.")
    parser.add_argument("--github_url", type=str, required=True, help="GitHub repository URL.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask.")
    parser.add_argument("--branch", type=str, default="main", help="Git branch.")
    parser.add_argument("--similarity_top_k", type=int, default=4, help="Top-k retrieved nodes.")
    parser.add_argument("--similarity_cutoff", type=float, default=0.7, help="Postprocessor cutoff.")
    parser.add_argument(
        "--response_mode",
        type=str,
        default="default",
        choices=["default", "compact", "tree_summarize", "no_text"],
        help="Response synthesis mode.",
    )
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
    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_mode=args.response_mode,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=args.similarity_cutoff)
        ],
    )

    response = query_engine.query(args.question)

    print("Question:")
    print(args.question)
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
