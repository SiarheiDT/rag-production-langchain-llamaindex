"""
Task 8: Add metadata to documents and inspect its effect before indexing/querying.

This script:
1. loads Wikipedia documents,
2. adds custom metadata,
3. builds an index,
4. runs a query,
5. prints answer + retrieved source chunks with metadata.

Usage:
    python m_02_008_metadata_demo.py
    python run.py --module 1 --part 2 --task 8
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.env_loader import load_env
from common.llama_settings import configure_llama
from llama_index.core import VectorStoreIndex, download_loader
from llama_index.core.node_parser import SentenceSplitter


def main() -> None:
    load_env()
    configure_llama()

    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    documents = loader.load_data(
        pages=["Natural Language Processing", "Artificial Intelligence"]
    )

    # Add custom metadata before indexing
    for i, doc in enumerate(documents, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        metadata["source_type"] = "wikipedia"
        metadata["document_order"] = i
        metadata["course_module"] = "LC_01_RAG_Foundations_Part_2"
        doc.metadata = metadata

    print("=== DOCUMENT METADATA PREVIEW ===")
    for i, doc in enumerate(documents, start=1):
        print(f"\nDocument {i}:")
        print(doc.metadata)

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
    )

    query_engine = index.as_query_engine(similarity_top_k=3)
    query = "What does NLP stand for?"
    response = query_engine.query(query)

    print("\n=== QUESTION ===")
    print(query)

    print("\n=== ANSWER ===")
    print(str(response))

    print("\n=== SOURCES WITH METADATA ===")
    for i, node in enumerate(response.source_nodes, start=1):
        print(f"\nSource {i} (score: {node.score}):")
        print("Metadata:")
        print(node.node.metadata)
        print("Text preview:")
        print(node.node.text[:300])


if __name__ == "__main__":
    main()