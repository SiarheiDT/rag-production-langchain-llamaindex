"""
llama_settings.py

Shared configuration for LlamaIndex across the project.
"""

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def build_llama_settings(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> None:
    """
    Configure global LlamaIndex settings.

    Parameters
    ----------
    model : str
        OpenAI model name.
    temperature : float
        Sampling temperature.
    """

    # LLM used for generation
    Settings.llm = OpenAI(
        model=model,
        temperature=temperature,
    )

    # Embedding model used for vector search
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small"
    )