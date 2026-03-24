"""
Example 2: Index a GitHub repository once and ask a single question.

This script is useful for quick smoke tests and CI-like execution.

Usage:
    python m_04_002_github_index_once.py --github_url https://github.com/owner/repo --question "How does the server work?"
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
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.deeplake import DeepLakeVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a GitHub repository and answer one question.")
    parser.add_argument("--github_url", type=str, required=True, help="GitHub repository URL.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask.")
    parser.add_argument("--dataset_path", type=str, default="./repository_db", help="Local or hub:// Deep Lake path.")
    parser.add_argument("--branch", type=str, default="main", help="Git branch.")
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

    vector_store = DeepLakeVectorStore(
        dataset_path=args.dataset_path,
        overwrite=True,
        runtime={"tensor_db": True} if str(args.dataset_path).startswith("hub://") else None,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    query_engine = index.as_query_engine()
    response = query_engine.query(args.question)

    print("Question:")
    print(args.question)
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
