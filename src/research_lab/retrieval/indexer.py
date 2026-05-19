from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.vector_stores.types import BasePydanticVectorStore


def build_index(
    data_dir: Path,
    vector_store: BasePydanticVectorStore,
) -> VectorStoreIndex:
    documents = SimpleDirectoryReader(input_dir=str(data_dir)).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)


def get_query_engine(index: VectorStoreIndex, top_k: int = 5) -> BaseQueryEngine:
    return index.as_query_engine(similarity_top_k=top_k)


if __name__ == "__main__":
    from research_lab.config import get_settings
    from research_lab.retrieval.pinecone_store import get_vector_store

    settings = get_settings()
    store = get_vector_store(settings)
    idx = build_index(Path(settings.data_dir), store)
    engine = get_query_engine(idx)
    result = engine.query("What is retrieval-augmented generation?")
    print(result)
