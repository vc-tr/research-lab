from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from research_lab.agents import (
    create_critic_agent,
    create_planner_agent,
    create_researcher_agent,
    create_writer_agent,
)
from research_lab.config import Settings
from research_lab.retrieval.web_search import WebSearchTool


class TestAgentCreation:
    def test_planner_agent_has_correct_role(self) -> None:
        agent = create_planner_agent(llm="anthropic/claude-sonnet-4-6")
        assert agent.role == "Research Planner"
        assert agent.allow_delegation is False

    def test_researcher_agent_has_tools(self) -> None:
        mock_tool = MagicMock(spec=WebSearchTool)
        mock_tool.name = "web_search"
        mock_tool.description = "Search the web"
        agent = create_researcher_agent(tools=[mock_tool], llm="anthropic/claude-sonnet-4-6")
        assert agent.role == "Research Analyst"
        assert agent.tools is not None
        assert len(agent.tools) == 1

    def test_critic_agent_has_correct_role(self) -> None:
        agent = create_critic_agent(llm="anthropic/claude-sonnet-4-6")
        assert agent.role == "Research Critic"

    def test_writer_agent_has_correct_role(self) -> None:
        agent = create_writer_agent(llm="anthropic/claude-sonnet-4-6")
        assert agent.role == "Technical Writer"


class TestCrewBuilding:
    @patch("research_lab.crew._build_researcher_tools")
    def test_build_crew_creates_correct_task_structure(self, mock_tools: Any) -> None:
        mock_tool = MagicMock(spec=WebSearchTool)
        mock_tool.name = "web_search"
        mock_tool.description = "Search"
        mock_tools.return_value = [mock_tool]

        from research_lab.crew import build_crew

        settings = Settings(use_pinecone=False, llm_provider="anthropic")
        crew = build_crew("What is RAG?", settings, max_subtasks=3)

        # 1 planner + 3 researchers + 1 critic + 1 writer = 6 agents
        assert len(crew.agents) == 6
        # 1 plan + 3 research + 1 critique + 1 write = 6 tasks
        assert len(crew.tasks) == 6

    @patch("research_lab.crew._build_researcher_tools")
    def test_research_tasks_have_async_execution(self, mock_tools: Any) -> None:
        mock_tool = MagicMock(spec=WebSearchTool)
        mock_tool.name = "web_search"
        mock_tool.description = "Search"
        mock_tools.return_value = [mock_tool]

        from research_lab.crew import build_crew

        settings = Settings(use_pinecone=False)
        crew = build_crew("Test question", settings, max_subtasks=2)

        # Tasks 1 and 2 (index 1, 2) are research tasks with async_execution
        research_tasks = crew.tasks[1:3]
        for task in research_tasks:
            assert task.async_execution is True

    @patch("research_lab.crew._build_researcher_tools")
    def test_step_callback_is_invoked(self, mock_tools: Any) -> None:
        mock_tool = MagicMock(spec=WebSearchTool)
        mock_tool.name = "web_search"
        mock_tool.description = "Search"
        mock_tools.return_value = [mock_tool]

        callback_calls: list[tuple[str, str, str]] = []

        def capture_callback(event_type: str, agent: str, content: str) -> None:
            callback_calls.append((event_type, agent, content))

        from research_lab.crew import build_crew

        settings = Settings(use_pinecone=False)
        crew = build_crew("Test", settings, step_callback=capture_callback, max_subtasks=2)

        # Simulate a task callback
        mock_output = MagicMock()
        mock_output.raw = "Some research findings"
        mock_output.agent = "Research Analyst"
        crew.tasks[0].callback(mock_output)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "agent_step"

    @patch("research_lab.crew._build_researcher_tools")
    def test_crew_task_order(self, mock_tools: Any) -> None:
        mock_tool = MagicMock(spec=WebSearchTool)
        mock_tool.name = "web_search"
        mock_tool.description = "Search"
        mock_tools.return_value = [mock_tool]

        from research_lab.crew import build_crew

        settings = Settings(use_pinecone=False)
        crew = build_crew("Test", settings, max_subtasks=2)

        tasks = crew.tasks
        plan_desc = tasks[0].description.lower()
        assert "decompose" in plan_desc or "subtask" in plan_desc
        for t in tasks[1:3]:
            desc = t.description.lower()
            assert "investigate" in desc or "subtask" in desc
        crit_desc = tasks[3].description.lower()
        assert "review" in crit_desc or "critique" in crit_desc
        write_desc = tasks[4].description.lower()
        assert "write" in write_desc or "report" in write_desc
