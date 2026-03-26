"""
m_02_006_langserve_export_stub.py

Purpose:
    Provide a minimal LangServe export stub for a LangChain chain.

What it demonstrates:
    - simple FastAPI app
    - add_routes from LangServe
    - exposing an LLM chain as an API

Notes:
    This is a deployment-oriented scaffold.
    It is intentionally small so that the user can understand the serving pattern
    without being distracted by application-specific logic.

Requirements:
    pip install fastapi uvicorn langserve langchain langchain-openai

Run:
    uvicorn m_02_006_langserve_export_stub:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

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

try:
    from fastapi import FastAPI
    from langserve import add_routes
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception as exc:
    raise SystemExit(
        f"Missing dependency for LangServe stub: {exc}\n"
        "Install with: pip install fastapi uvicorn langserve langchain langchain-openai"
    )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise assistant."),
        ("human", "Explain in 4 lines how LangServe fits into the LangChain ecosystem."),
    ]
)

chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()

app = FastAPI(title="LangServe Export Stub", version="0.1.0")
add_routes(app, chain, path="/explain")
