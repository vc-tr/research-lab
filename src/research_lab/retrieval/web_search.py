from __future__ import annotations

from typing import Any

from crewai.tools import BaseTool
from langchain_core.tools import tool as langchain_tool
from pydantic import BaseModel, Field


def _get_search_backend() -> Any:
    try:
        from langchain_community.tools import TavilySearchResults

        return TavilySearchResults(max_results=3)
    except (ImportError, Exception):
        from langchain_community.tools import DuckDuckGoSearchResults

        return DuckDuckGoSearchResults(max_results=3)


_backend: Any = None


def _backend_instance() -> Any:
    global _backend  # noqa: PLW0603
    if _backend is None:
        _backend = _get_search_backend()
    return _backend


@langchain_tool  # type: ignore[arg-type]
def web_search_langchain(query: str) -> str:
    """Search the web for recent information on a topic. Returns top 3 results with sources."""
    backend = _backend_instance()
    results: str = backend.run(query)
    return results


class _WebSearchInput(BaseModel):
    query: str = Field(description="The search query")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for recent information. Returns top results with sources."
    args_schema: type[BaseModel] = _WebSearchInput  # type: ignore[assignment]

    def _run(self, query: str) -> str:
        backend = _backend_instance()
        result: str = backend.run(query)
        return result


def get_web_search_tool() -> WebSearchTool:
    return WebSearchTool()
