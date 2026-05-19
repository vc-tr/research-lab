from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from crewai import Crew, Process, Task
from crewai.tools import BaseTool

from research_lab.agents import (
    create_critic_agent,
    create_planner_agent,
    create_researcher_agent,
    create_writer_agent,
)
from research_lab.config import Settings
from research_lab.retrieval.pinecone_store import get_vector_store
from research_lab.retrieval.web_search import get_web_search_tool


def _export_llm_env(settings: Settings) -> None:
    """LiteLLM reads provider keys from os.environ; export from settings."""
    if settings.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.groq_api_key:
        os.environ["GROQ_API_KEY"] = settings.groq_api_key


def _get_llm_string(settings: Settings) -> str:
    _export_llm_env(settings)
    if settings.llm_provider == "openai":
        return f"openai/{settings.default_model}"
    if settings.llm_provider == "groq":
        return f"groq/{settings.default_model}"
    return f"anthropic/{settings.default_model}"


def _build_researcher_tools(settings: Settings) -> list[BaseTool]:
    tools: list[BaseTool] = [get_web_search_tool()]

    data_path = Path(settings.data_dir)
    if data_path.exists() and any(data_path.glob("*.pdf")):
        from research_lab.retrieval.indexer import build_index
        from research_lab.retrieval.query_tool import PaperQueryTool

        store = get_vector_store(settings)
        index = build_index(data_path, store)
        tools.append(PaperQueryTool(index=index))

    return tools


def build_crew(
    question: str,
    settings: Settings,
    step_callback: Callable[[str, str, str], None] | None = None,
    max_subtasks: int = 5,
) -> Crew:
    llm = _get_llm_string(settings)

    planner = create_planner_agent(llm=llm)
    critic = create_critic_agent(llm=llm)
    writer = create_writer_agent(llm=llm)

    researcher_tools = _build_researcher_tools(settings)

    def _task_callback(output: Any) -> None:
        if step_callback and hasattr(output, "raw"):
            agent_name = getattr(output, "agent", "unknown")
            step_callback("agent_step", str(agent_name), str(output.raw)[:500])

    plan_task = Task(
        description=(
            f"Analyze the following research question and decompose it into "
            f"2-{max_subtasks} focused subtasks:\n\n{question}\n\n"
            f"Return a numbered list of subtasks, each on its own line."
        ),
        expected_output="A numbered list of 2-5 research subtasks.",
        agent=planner,
        callback=_task_callback,
    )

    researchers: list[Any] = []
    research_tasks: list[Task] = []
    for i in range(max_subtasks):
        researcher = create_researcher_agent(tools=researcher_tools, llm=llm)
        researchers.append(researcher)

        task = Task(
            description=(
                f"Investigate subtask #{i + 1} from the research plan. "
                f"Use the paper_search tool and web_search tool to gather evidence. "
                f"Provide detailed findings with inline citations."
            ),
            expected_output=(
                "Detailed research findings with citations in the format [source_name, page/URL]."
            ),
            agent=researcher,
            context=[plan_task],
            async_execution=True,
            callback=_task_callback,
        )
        research_tasks.append(task)

    critique_task = Task(
        description=(
            "Review all research findings from the analysts. "
            "Identify unsupported claims, logical gaps, and missing evidence. "
            "Provide specific, actionable feedback."
        ),
        expected_output=(
            "A critique listing: (1) claims that need better evidence, "
            "(2) gaps in coverage, (3) logical inconsistencies."
        ),
        agent=critic,
        context=research_tasks,
        callback=_task_callback,
    )

    write_task = Task(
        description=(
            f"Write a comprehensive Markdown report answering: {question}\n\n"
            f"Incorporate the research findings and address the critic's feedback. "
            f"Include:\n"
            f"- A clear title and introduction\n"
            f"- Well-organized sections with headings\n"
            f"- Inline citations [1], [2], etc.\n"
            f"- A ## References section at the end with full citation details"
        ),
        expected_output="A complete Markdown research report with citations and references.",
        agent=writer,
        context=[critique_task, *research_tasks],
        callback=_task_callback,
    )

    all_agents = [planner, *researchers, critic, writer]
    all_tasks = [plan_task, *research_tasks, critique_task, write_task]

    return Crew(
        agents=all_agents,
        tasks=all_tasks,
        process=Process.sequential,
        manager_llm=llm,
        verbose=True,
    )


def run_research(
    question: str,
    settings: Settings | None = None,
    step_callback: Callable[[str, str, str], None] | None = None,
) -> str:
    if settings is None:
        from research_lab.config import get_settings

        settings = get_settings()

    crew = build_crew(question, settings, step_callback=step_callback)
    result = crew.kickoff()
    return str(result)
