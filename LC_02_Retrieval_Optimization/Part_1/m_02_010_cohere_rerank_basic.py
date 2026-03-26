"""
m_02_010_cohere_rerank_basic.py

Runs the standalone Cohere rerank example from the module.

Usage:
    python m_02_010_cohere_rerank_basic.py
    python m_02_010_cohere_rerank_basic.py --top-n 3 --model rerank-v3.5
"""

from __future__ import annotations

import argparse
import os

import cohere

from m_02_003_common import require_env
from common.env_loader import load_env

DEFAULT_DOCUMENTS = [
    "Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
    "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
    "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
    "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
    "North Dakota is a state in the United States. 672,591 people lived in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone Cohere rerank example.")
    parser.add_argument("--query", default="What is the capital of the United States?", help="Query to rerank against.")
    parser.add_argument("--top-n", type=int, default=3, help="How many top results to return.")
    parser.add_argument("--model", default="rerank-v3.5", help="Cohere rerank model.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    load_env()
    require_env("COHERE_API_KEY")
    client = cohere.Client(os.environ["COHERE_API_KEY"])

    response = client.rerank(
        query=args.query,
        documents=DEFAULT_DOCUMENTS,
        top_n=args.top_n,
        model=args.model,
    )

    for rank, item in enumerate(response.results, start=1):
        doc_text = DEFAULT_DOCUMENTS[item.index]
        score = item.relevance_score
        print(f"Document Rank: {rank}")
        print(f"Document: {doc_text}")
        print(f"Relevance Score: {score:.2f}")
        print()


if __name__ == "__main__":
    main(parse_args())
