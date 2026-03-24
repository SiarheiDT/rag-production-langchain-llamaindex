from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

def configure_llama():
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small"
    )