from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from research_lab.api.app import app
from research_lab.api.schemas import StreamEvent


def _fake_run_crew_sync(
    question: str,
    max_subtasks: int,
    queue: asyncio.Queue[StreamEvent],
    loop: asyncio.AbstractEventLoop,
) -> None:
    events = [
        StreamEvent(type="agent_step", agent="Research Planner", content="Planning subtasks..."),
        StreamEvent(type="tool_call", agent="Research Analyst", content="web_search('RAG')"),
        StreamEvent(type="agent_step", agent="Research Critic", content="Reviewing findings..."),
        StreamEvent(
            type="final",
            content="# Report\n\nFindings here.\n\n## References\n[1] Source A",
        ),
    ]
    for event in events:
        loop.call_soon_threadsafe(queue.put_nowait, event)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class TestHealthEndpoint:
    @pytest.mark.anyio
    async def test_health_returns_ok(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestResearchEndpoint:
    @pytest.mark.anyio
    @patch("research_lab.api.app._run_crew_sync", side_effect=_fake_run_crew_sync)
    async def test_research_streams_sse_events(self, mock_run: Any) -> None:
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://test") as client,
            client.stream(
                "POST",
                "/research",
                json={"question": "What is RAG?"},
            ) as resp,
        ):
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            events: list[dict[str, Any]] = []
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    events.append(data)

            event_types = [e["type"] for e in events]
            assert "agent_step" in event_types
            assert "tool_call" in event_types
            assert "final" in event_types

            final = next(e for e in events if e["type"] == "final")
            assert "# Report" in final["content"]
            assert "References" in final["content"]

    @pytest.mark.anyio
    async def test_research_rejects_empty_question(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/research", json={"question": ""})
            assert resp.status_code == 422
