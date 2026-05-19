from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from research_lab.api.schemas import ResearchRequest, StreamEvent
from research_lab.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from research_lab.observability.langsmith_setup import configure_langsmith

    configure_langsmith(get_settings())
    yield


app = FastAPI(title="Research Lab", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _run_crew_sync(
    question: str,
    max_subtasks: int,
    queue: asyncio.Queue[StreamEvent],
    loop: asyncio.AbstractEventLoop,
) -> None:
    from research_lab.config import get_settings
    from research_lab.crew import build_crew

    settings = get_settings()

    def step_callback(event_type: str, agent: str, content: str) -> None:
        event = StreamEvent(type=event_type, agent=agent, content=content)  # type: ignore[arg-type]
        loop.call_soon_threadsafe(queue.put_nowait, event)

    crew = build_crew(question, settings, step_callback=step_callback, max_subtasks=max_subtasks)

    try:
        result = crew.kickoff()
        final_event = StreamEvent(type="final", content=str(result))
        loop.call_soon_threadsafe(queue.put_nowait, final_event)
    except Exception as e:
        error_event = StreamEvent(type="final", content=f"Error: {e}")
        loop.call_soon_threadsafe(queue.put_nowait, error_event)


@app.post("/research")
async def research(request: ResearchRequest) -> StreamingResponse:
    run_id = str(uuid.uuid4())
    queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    asyncio.get_running_loop().run_in_executor(
        None, _run_crew_sync, request.question, request.max_subtasks, queue, loop
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            event = await queue.get()
            yield f"data: {event.model_dump_json()}\n\n"
            if event.type == "final":
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Run-ID": run_id, "Cache-Control": "no-cache"},
    )
