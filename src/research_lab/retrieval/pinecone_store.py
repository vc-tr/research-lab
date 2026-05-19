from __future__ import annotations

from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from research_lab.config import Settings


def get_vector_store(settings: Settings) -> BasePydanticVectorStore:
    if not settings.use_pinecone:
        return SimpleVectorStore()

    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from pinecone import Pinecone as PineconeClient

    pc = PineconeClient(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    return PineconeVectorStore(pinecone_index=index)
