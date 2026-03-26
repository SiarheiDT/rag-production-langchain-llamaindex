"""
m_02_004_langsmith_traced_rag_pipeline.py

Purpose:
    Build a local RAG pipeline with:
    - FAISS vector store
    - MMR retrieval
    - Cross-encoder reranking
    - LangSmith tracing
    - Debug visibility (critical for interviews)

Requirements:
    pip install langchain langchain-openai langchain-community faiss-cpu langchain-text-splitters sentence-transformers

Run:
    python m_02_004_langsmith_traced_rag_pipeline.py \
        --docs-dir ./data/sample_docs
"""

from __future__ import annotations

import argparse
from pathlib import Path


# =========================
# ENV LOADING
# =========================
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


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production-like traced RAG pipeline")
    parser.add_argument("--docs-dir", type=Path, required=True)
    parser.add_argument(
        "--question",
        type=str,
        default="What is the main idea of these documents?",
    )
    return parser.parse_args()


# =========================
# LOAD DATA
# =========================
def load_texts(docs_dir: Path) -> list[str]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Directory not found: {docs_dir}")

    texts = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            texts.append(path.read_text(encoding="utf-8"))

    if not texts:
        raise ValueError(f"No supported files found under: {docs_dir}")

    return texts


# =========================
# MAIN
# =========================
def main() -> int:
    args = parse_args()

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        print("Missing dependency.")
        print(f"Details: {exc}")
        return 1

    # =========================
    # LOAD + CHUNK
    # =========================
    raw_texts = load_texts(args.docs_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []
    for text in raw_texts:
        chunks.extend(splitter.split_text(text))

    print(f"Loaded {len(chunks)} chunks")

    # =========================
    # VECTOR STORE
    # =========================
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        # search_kwargs={"k": 5, "lambda_mult": 0.7}
        search_kwargs={"k": 4, "lambda_mult": 0.7}
    )

    # =========================
    # RETRIEVE
    # =========================
    docs = retriever.invoke(args.question)

    print("\n=== BEFORE RERANK ===")
    for i, doc in enumerate(docs):
        print(f"\n--- DOC {i+1} ---")
        print(doc.page_content[:300])

    # =========================
    # RERANK (CRITICAL)
    # =========================
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(args.question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]

    print("\n=== AFTER RERANK ===")
    for i, doc in enumerate(docs):
        print(f"\n--- DOC {i+1} ---")
        print(doc.page_content[:300])

    # =========================
    # BUILD CONTEXT
    # =========================
    context = "\n\n".join(doc.page_content for doc in docs[:2] if "RAG" in doc.page_content or "Evaluation" in doc.page_content)
    

    # =========================
    # PROMPT
    # =========================
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer ONLY using the provided context. "
                "If multiple topics exist, focus on the most relevant one. "
                "Ignore unrelated information. "
                "Do not hallucinate."
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}"
            ),
        ]
    )

    # =========================
    # MODEL
    # =========================
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    answer = chain.invoke({
        "question": args.question,
        "context": context
    })

    # =========================
    # OUTPUT
    # =========================
    print("\n" + "=" * 88)
    print("PRODUCTION-LIKE RAG PIPELINE")
    print("=" * 88)
    print(f"Question: {args.question}")
    print("-" * 88)
    print("Final context:")
    print(context[:1000])
    print("-" * 88)
    print("Answer:")
    print(answer)
    print("=" * 88)

    print("Check LangSmith for full trace (retrieval + rerank + generation)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())