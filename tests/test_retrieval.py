from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from llama_index.core import Document

from research_lab.config import Settings
from research_lab.retrieval.indexer import build_index, get_query_engine
from research_lab.retrieval.pinecone_store import get_vector_store


class TestPineconeStore:
    def test_returns_simple_store_when_pinecone_disabled(self) -> None:
        settings = Settings(use_pinecone=False)
        store = get_vector_store(settings)
        assert store is not None
        assert type(store).__name__ == "SimpleVectorStore"

    def test_returns_pinecone_store_when_enabled(self) -> None:
        with (
            patch("pinecone.Pinecone") as mock_pc_cls,
            patch("llama_index.vector_stores.pinecone.PineconeVectorStore") as mock_pvs_cls,
        ):
            mock_index = MagicMock()
            mock_pc_cls.return_value.Index.return_value = mock_index
            mock_pvs_cls.return_value = MagicMock()

            settings = Settings(
                use_pinecone=True, pinecone_api_key="test", pinecone_index_name="test-idx"
            )
            store = get_vector_store(settings)

            mock_pc_cls.assert_called_once_with(api_key="test")
            mock_pc_cls.return_value.Index.assert_called_once_with("test-idx")
            assert store is not None


class TestIndexer:
    @patch("research_lab.retrieval.indexer.SimpleDirectoryReader")
    def test_build_index_creates_queryable_index(self, mock_reader_cls: Any) -> None:
        mock_docs = [
            Document(text="RAG combines retrieval with generation."),
            Document(text="Vector databases store embeddings efficiently."),
        ]
        mock_reader_cls.return_value.load_data.return_value = mock_docs

        settings = Settings(use_pinecone=False)
        store = get_vector_store(settings)
        index = build_index(Path("data/papers"), store)

        assert index is not None
        engine = get_query_engine(index, top_k=1)
        assert engine is not None


class TestWebSearch:
    def test_web_search_tool_returns_results(self) -> None:
        import research_lab.retrieval.web_search as ws

        mock_backend = MagicMock()
        mock_backend.run.return_value = "[1] Example result - https://example.com"
        ws._backend = mock_backend

        tool = ws.get_web_search_tool()
        result = tool.run(query="test query")
        assert isinstance(result, str)
        mock_backend.run.assert_called_once_with("test query")

        ws._backend = None

    def test_get_web_search_tool_is_crewai_base_tool(self) -> None:
        from crewai.tools import BaseTool

        import research_lab.retrieval.web_search as ws

        ws._backend = MagicMock()
        tool = ws.get_web_search_tool()
        assert isinstance(tool, BaseTool)
        ws._backend = None
