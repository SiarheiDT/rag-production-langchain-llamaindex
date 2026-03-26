"""
Example 1: Chat with a GitHub repository using LlamaIndex + Deep Lake.

This is the main high-level script for the module.
It loads a GitHub repository, stores it in a Deep Lake vector store,
builds a LlamaIndex VectorStoreIndex, and starts an interactive Q&A loop.

Usage examples:
    python m_04_001_github_quickstart.py --github_url http://github.com/langchain-ai/langchain-academy
    python m_04_001_github_quickstart.py --github_url https://github.com/owner/repo --dataset_path ./repository_db
    python m_04_001_github_quickstart.py --github_url https://github.com/owner/repo --dataset_path hub://YOUR_ORG/repository_vector_store
"""

import argparse
import os
import re
import sys
import textwrap
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
    parser = argparse.ArgumentParser(description="Index a GitHub repository and chat with its code.")
    parser.add_argument("--github_url", type=str, required=True, help="GitHub repository URL.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./repository_db",
        help="Deep Lake dataset path. Use a local directory or a hub:// path.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Git branch to load.",
    )
    parser.add_argument(
        "--include_ext",
        nargs="+",
        default=[".py", ".js", ".ts", ".md"],
        help="File extensions to include.",
    )
    parser.add_argument(
        "--concurrent_requests",
        type=int,
        default=5,
        help="Number of concurrent GitHub requests.",
    )
    parser.add_argument(
        "--intro_question",
        type=str,
        default="What is the repository about?",
        help="Test question to ask after indexing.",
    )
    return parser.parse_args()


def parse_github_url(url: str) -> tuple[str | None, str | None]:
    pattern = r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$"
    match = re.match(pattern, url.strip())
    return match.groups() if match else (None, None)


def validate_owner_repo(owner: str | None, repo: str | None) -> bool:
    return bool(owner) and bool(repo)


def initialize_github_client() -> GithubClient:
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise EnvironmentError("GITHUB_TOKEN not found in environment variables.")
    return GithubClient(github_token=github_token)


def build_loader(
    github_client: GithubClient,
    owner: str,
    repo: str,
    include_ext: list[str],
    concurrent_requests: int,
) -> GithubRepositoryReader:
    return GithubRepositoryReader(
        github_client=github_client,
        owner=owner,
        repo=repo,
        filter_file_extensions=(
            include_ext,
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        verbose=False,
        concurrent_requests=concurrent_requests,
    )


def main() -> None:
    nest_asyncio.apply()
    load_env()
    configure_llama()
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")
    if not os.getenv("ACTIVELOOP_TOKEN") and str(args.dataset_path).startswith("hub://"):
        raise EnvironmentError("ACTIVELOOP_TOKEN not found in environment variables for hub:// dataset.")

    owner, repo = parse_github_url(args.github_url)
    if not validate_owner_repo(owner, repo):
        raise ValueError("Invalid GitHub URL.")

    github_client = initialize_github_client()
    loader = build_loader(
        github_client=github_client,
        owner=owner,
        repo=repo,
        include_ext=args.include_ext,
        concurrent_requests=args.concurrent_requests,
    )

    print(f"Loading repository '{repo}' by '{owner}' from branch '{args.branch}'...")
    docs = loader.load_data(branch=args.branch)

    print("\nDocuments uploaded:")
    for doc in docs[:20]:
        print(doc.metadata)

    print("\nUploading to vector store...")
    vector_store = DeepLakeVectorStore(
        dataset_path=args.dataset_path,
        overwrite=True,
        runtime={"tensor_db": True} if str(args.dataset_path).startswith("hub://") else None,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    query_engine = index.as_query_engine()

    print(f"\nTest question: {args.intro_question}")
    print("=" * 50)
    answer = query_engine.query(args.intro_question)
    print(f"Answer: {textwrap.fill(str(answer), 100)}\n")

    while True:
        user_question = input("Please enter your question (or type 'exit' to quit): ").strip()
        if user_question.lower() == "exit":
            print("Exiting, thanks for chatting!")
            break

        print(f"\nYour question: {user_question}")
        print("=" * 50)
        answer = query_engine.query(user_question)
        print(f"Answer: {textwrap.fill(str(answer), 100)}\n")


if __name__ == "__main__":
    main()
