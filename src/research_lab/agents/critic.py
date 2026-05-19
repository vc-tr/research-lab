from __future__ import annotations

from crewai import Agent


def create_critic_agent(llm: str | None = None) -> Agent:
    return Agent(
        role="Research Critic",
        goal=(
            "Critically evaluate research findings for accuracy, completeness, "
            "and logical consistency. Identify unsupported claims and gaps in evidence."
        ),
        backstory=(
            "You are a rigorous peer reviewer with expertise in evaluating research "
            "quality. You challenge weak claims, flag missing evidence, and ensure "
            "that conclusions are well-supported by the gathered data."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
