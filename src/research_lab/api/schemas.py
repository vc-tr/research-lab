from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    max_subtasks: int = Field(default=5, ge=2, le=10)


class StreamEvent(BaseModel):
    type: Literal["agent_step", "tool_call", "final"] = "agent_step"
    agent: str | None = None
    content: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResearchResponse(BaseModel):
    run_id: str
    report: str
    sources: list[str] = Field(default_factory=list)
