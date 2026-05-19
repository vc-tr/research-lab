from __future__ import annotations

from crewai import Agent
from crewai.tools import BaseTool


def create_researcher_agent(
    tools: list[BaseTool],
    llm: str | None = None,
) -> Agent:
    return Agent(
        role="Research Analyst",
        goal=(
            "Investigate a specific subtask thoroughly using available retrieval "
            "and web search tools. Provide detailed findings with inline citations."
        ),
        backstory=(
            "You are a meticulous research analyst who combines academic paper "
            "retrieval with web search to gather comprehensive, well-sourced evidence. "
            "You always cite your sources with titles and URLs."
        ),
        tools=tools,
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
