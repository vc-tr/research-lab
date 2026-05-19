from __future__ import annotations

from crewai import Agent


def create_planner_agent(llm: str | None = None) -> Agent:
    return Agent(
        role="Research Planner",
        goal=(
            "Decompose a complex research question into 2-5 focused, "
            "non-overlapping subtasks that together cover the full scope of the question."
        ),
        backstory=(
            "You are a senior research strategist who excels at breaking down "
            "complex questions into manageable investigations. You identify the "
            "key dimensions of a topic and ensure comprehensive coverage."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
