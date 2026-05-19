from __future__ import annotations

from typing import Any

from crewai.tools import BaseTool
from llama_index.core import VectorStoreIndex
from pydantic import BaseModel, Field


class _QueryInput(BaseModel):
    query: str = Field(description="The search query to run against the paper corpus")


class PaperQueryTool(BaseTool):
    name: str = "paper_search"
    description: str = (
        "Search the local corpus of academic papers (arXiv PDFs) for relevant information. "
        "Use this for finding technical details, definitions, and research findings."
    )
    args_schema: type[BaseModel] = _QueryInput
    _query_engine: Any = None

    def __init__(self, index: VectorStoreIndex, top_k: int = 5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._query_engine = index.as_query_engine(similarity_top_k=top_k)

    def _run(self, query: str) -> str:
        response = self._query_engine.query(query)
        sources: list[str] = []
        for node in response.source_nodes:
            meta = node.metadata
            filename = meta.get("file_name", "unknown")
            page = meta.get("page_label", "?")
            sources.append(f"[{filename}, p.{page}]")

        source_str = ", ".join(sources) if sources else "no sources"
        return f"{response}\n\nSources: {source_str}"
