"""
m_02_003_common.py

Shared helper functions for Module 2 examples. These helpers keep the example
scripts focused while still making them runnable as standalone files.

Expected environment variables:
    OPENAI_API_KEY
    ACTIVELOOP_TOKEN
    COHERE_API_KEY  # only for rerank scripts
"""

from __future__ import annotations

import os
import pathlib
from typing import List, Tuple

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.deeplake import DeepLakeVectorStore


DEFAULT_DATA_DIR = pathlib.Path("./data/paul_graham")
DEFAULT_DATA_FILE = DEFAULT_DATA_DIR / "paul_graham_essay.txt"


def require_env(var_name: str) -> str:
    """Return an environment variable or raise a descriptive error."""
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"Environment variable {var_name} is required but not set.")
    return value


def load_documents(data_dir: str | pathlib.Path = DEFAULT_DATA_DIR):
    """Load documents from a directory using LlamaIndex SimpleDirectoryReader."""
    return SimpleDirectoryReader(str(data_dir)).load_data()


def build_nodes(
    documents,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    """Split documents into nodes using LlamaIndex Settings.node_parser."""
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    node_parser = Settings.node_parser
    return node_parser.get_nodes_from_documents(documents)


def create_deeplake_vector_store(
    org_id: str,
    dataset_name: str = "LlamaIndex_paulgraham_essay",
    overwrite: bool = False,
):
    """Create a Deep Lake vector store configured for the given organization."""
    require_env("ACTIVELOOP_TOKEN")
    dataset_path = f"hub://{org_id}/{dataset_name}"
    vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=overwrite)
    return vector_store, dataset_path


def build_vector_index(
    org_id: str,
    dataset_name: str = "LlamaIndex_paulgraham_essay",
    data_dir: str | pathlib.Path = DEFAULT_DATA_DIR,
    overwrite: bool = False,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Tuple[VectorStoreIndex, str, List]:
    """
    End-to-end helper:
    1. load source documents
    2. split into nodes
    3. create Deep Lake vector store
    4. create storage context
    5. build VectorStoreIndex
    """
    documents = load_documents(data_dir=data_dir)
    nodes = build_nodes(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vector_store, dataset_path = create_deeplake_vector_store(
        org_id=org_id,
        dataset_name=dataset_name,
        overwrite=overwrite,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(nodes)

    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index, dataset_path, nodes
