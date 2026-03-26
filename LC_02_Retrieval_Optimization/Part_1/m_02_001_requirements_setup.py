"""
m_02_001_requirements_setup.py

Installs the core dependencies used across Module 2 examples and validates that
required API keys are available in the environment.

Usage:
    python m_02_001_requirements_setup.py --install
    python m_02_001_requirements_setup.py --check-env

Expected environment variables:
    OPENAI_API_KEY
    ACTIVELOOP_TOKEN
    COHERE_API_KEY  # only needed for rerank examples
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Iterable


REQUIRED_PACKAGES = [
    "openai>=1.0.0",
    "tiktoken",
    "llama-index>=0.10.0",
    "llama-index-vector-stores-deeplake",
    "llama-index-embeddings-openai",
    "llama-index-llms-openai",
    "llama-index-postprocessor-cohere-rerank",
    "llama-index-question-gen-openai",
    "deeplake>=4.0.0",
    "langchain>=0.1.0",
    "cohere",
]


def install_packages(packages: Iterable[str]) -> None:
    """Install Python packages using the current interpreter."""
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def check_env() -> None:
    """Validate the API keys required by this module."""
    keys = ["OPENAI_API_KEY", "ACTIVELOOP_TOKEN", "COHERE_API_KEY"]
    for key in keys:
        value = os.getenv(key)
        print(f"{key}: {'SET' if value else 'MISSING'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install dependencies or check API keys for Module 2.")
    parser.add_argument("--install", action="store_true", help="Install required Python packages.")
    parser.add_argument("--check-env", action="store_true", help="Check required environment variables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.install and not args.check_env:
        raise SystemExit("Specify at least one action: --install and/or --check-env")

    if args.install:
        install_packages(REQUIRED_PACKAGES)

    if args.check_env:
        check_env()


if __name__ == "__main__":
    main()
