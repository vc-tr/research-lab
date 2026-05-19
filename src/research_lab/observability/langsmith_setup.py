from __future__ import annotations

import os
import uuid

from research_lab.config import Settings


def configure_langsmith(settings: Settings) -> None:
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


def tag_run(run_id: str | None = None) -> dict[str, str]:
    if run_id is None:
        run_id = str(uuid.uuid4())
    return {
        "run_id": run_id,
        "project": "research-lab",
    }
