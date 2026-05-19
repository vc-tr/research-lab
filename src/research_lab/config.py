from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: Literal["anthropic", "openai", "groq"] = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    default_model: str = "claude-sonnet-4-6"

    # Pinecone
    use_pinecone: bool = True
    pinecone_api_key: str = ""
    pinecone_index_name: str = "research-lab"

    # Search
    tavily_api_key: str = ""

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "research-lab"

    # Data
    data_dir: str = "data/papers"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
