"""CrewAI agent definitions: Planner, Researcher, Critic, Writer."""

from research_lab.agents.critic import create_critic_agent
from research_lab.agents.planner import create_planner_agent
from research_lab.agents.researcher import create_researcher_agent
from research_lab.agents.writer import create_writer_agent

__all__ = [
    "create_critic_agent",
    "create_planner_agent",
    "create_researcher_agent",
    "create_writer_agent",
]
