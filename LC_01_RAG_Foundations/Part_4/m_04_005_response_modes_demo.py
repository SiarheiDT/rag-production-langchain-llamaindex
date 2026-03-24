"""
Example 5: Compare LlamaIndex response modes.

This script runs the same question with different response modes:
default, compact, tree_summarize, no_text.

Usage:
    python m_04_005_response_modes_demo.py --github_url https://github.com/owner/repo --question "What is the repository about?"
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
    parser = argparse.ArgumentParser(description="Compare response modes on a GitHub repository index.")
    parser.add_argument("--github_url", type=str, required=True, help="GitHub repository URL.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask.")
    parser.add_argument("--branch", type=str, default="main", help="Git branch.")
    parser.add_argument("--similarity_top_k", type=int, default=4, help="Top-k retrieved nodes.")
    parser.add_argument("--similarity_cutoff", type=float, default=0.7, help="Similarity cutoff.")
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

    modes = ["default", "compact", "tree_summarize", "no_text"]

    for mode in modes:
        print("\n" + "=" * 80)
        print(f"response_mode={mode}")

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_mode=mode,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=args.similarity_cutoff)
            ],
        )

        response = query_engine.query(args.question)
        print(response)


if __name__ == "__main__":
    main()
