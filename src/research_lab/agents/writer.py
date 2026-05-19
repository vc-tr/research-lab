from __future__ import annotations

from crewai import Agent


def create_writer_agent(llm: str | None = None) -> Agent:
    return Agent(
        role="Technical Writer",
        goal=(
            "Synthesize research findings into a well-structured Markdown report "
            "with proper citations, clear headings, and a References section."
        ),
        backstory=(
            "You are an expert technical writer who transforms raw research into "
            "polished, reader-friendly reports. You organize information logically, "
            "write clear prose, and always include a complete References section "
            "with numbered citations."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
