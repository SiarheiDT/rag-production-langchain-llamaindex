"""
m_02_001_langsmith_env_check.py

Purpose:
    Validate that the minimum environment required for LangSmith tracing is present.

What it checks:
    - OPENAI_API_KEY
    - LANGSMITH_API_KEY
    - LANGSMITH_TRACING_V2
    - LANGSMITH_PROJECT
    - LANGSMITH_ENDPOINT

Why this matters:
    Before running traced RAG / LangChain examples, it is useful to verify that
    the environment is wired correctly. This script is intentionally simple and
    safe to run first.

Run:
    python m_02_001_langsmith_env_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from common.env_loader import load_env
    load_env()
except Exception:
    try:
        from dotenv import load_dotenv
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")
    except Exception:
        pass


def mask(value: str | None, keep: int = 6) -> str:
    """Mask secrets while still showing whether they were loaded."""
    if not value:
        return "MISSING"
    if len(value) <= keep:
        return "*" * len(value)
    return value[:keep] + "..."


def main() -> int:
    required = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY"),
        "LANGSMITH_TRACING_V2": os.getenv("LANGSMITH_TRACING_V2"),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT"),
        "LANGSMITH_ENDPOINT": os.getenv("LANGSMITH_ENDPOINT"),
    }

    print("=" * 88)
    print("LangSmith Environment Check")
    print("=" * 88)

    missing = []
    for key, value in required.items():
        display = value if key in {"LANGSMITH_TRACING_V2", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"} else mask(value)
        print(f"{key:24s}: {display}")
        if not value:
            missing.append(key)

    print("-" * 88)

    if missing:
        print("Status: FAILED")
        print("Missing variables:")
        for item in missing:
            print(f"  - {item}")
        return 1

    tracing_enabled = str(required["LANGSMITH_TRACING_V2"]).lower() == "true"
    if not tracing_enabled:
        print("Status: WARNING")
        print("LANGSMITH_TRACING_V2 is present but not set to 'true'.")
        return 1

    print("Status: OK")
    print("Environment looks ready for LangSmith tracing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
