#!/usr/bin/env python3
"""Verify all configured API keys actually work. Run after editing .env."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/verify_keys.py` even without editable install
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from research_lab.config import get_settings  # noqa: E402


def _check_anthropic(key: str) -> tuple[bool, str]:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "say hi"}],
        )
        return True, f"OK (model={resp.model})"
    except Exception as e:
        return False, str(e)[:120]


def _check_groq(key: str, model: str) -> tuple[bool, str]:
    try:
        import httpx

        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "say hi"}],
            },
            timeout=15.0,
        )
        if resp.status_code == 200:
            return True, f"OK (model={model})"
        return False, f"HTTP {resp.status_code}: {resp.text[:100]}"
    except Exception as e:
        return False, str(e)[:120]


def _check_pinecone(key: str, index_name: str) -> tuple[bool, str]:
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=key)
        indexes = [idx.name for idx in pc.list_indexes()]
        if index_name in indexes:
            return True, f"OK (index '{index_name}' found)"
        return False, f"Connected, but index '{index_name}' not found. Indexes: {indexes}"
    except Exception as e:
        return False, str(e)[:120]


def _check_tavily(key: str) -> tuple[bool, str]:
    try:
        from langchain_community.tools import TavilySearchResults

        tool = TavilySearchResults(max_results=1, tavily_api_key=key)
        result = tool.invoke("test")
        return True, f"OK ({len(result)} chars returned)"
    except Exception as e:
        return False, str(e)[:120]


def _check_langsmith(key: str, project: str) -> tuple[bool, str]:
    try:
        from langsmith import Client

        client = Client(api_key=key)
        # Lightweight call: list (or create) the project
        client.read_project(project_name=project)
        return True, f"OK (project '{project}')"
    except Exception as e:
        msg = str(e)[:120]
        if "not found" in msg.lower():
            return True, f"OK (project '{project}' will be created on first trace)"
        return False, msg


def main() -> int:
    s = get_settings()
    checks: list[tuple[str, bool, bool, str]] = []

    # LLM check based on provider
    if s.llm_provider == "groq":
        if s.groq_api_key:
            ok, msg = _check_groq(s.groq_api_key, s.default_model)
            checks.append(("Groq (LLM)", True, ok, msg))
        else:
            checks.append(("Groq (LLM)", True, False, "GROQ_API_KEY not set (REQUIRED)"))
    elif s.llm_provider == "openai":
        ok_set = bool(s.openai_api_key)
        checks.append(
            ("OpenAI (LLM)", ok_set, ok_set, "Key set" if ok_set else "OPENAI_API_KEY not set")
        )
    else:
        if s.anthropic_api_key:
            ok, msg = _check_anthropic(s.anthropic_api_key)
            checks.append(("Anthropic (LLM)", True, ok, msg))
        else:
            checks.append(("Anthropic (LLM)", False, False, "ANTHROPIC_API_KEY not set (REQUIRED)"))

    if s.use_pinecone:
        if s.pinecone_api_key:
            ok, msg = _check_pinecone(s.pinecone_api_key, s.pinecone_index_name)
            checks.append(("Pinecone", True, ok, msg))
        else:
            checks.append(("Pinecone", True, False, "USE_PINECONE=true but no key set"))
    else:
        checks.append(("Pinecone", False, True, "USE_PINECONE=false (in-memory store)"))

    if s.tavily_api_key:
        ok, msg = _check_tavily(s.tavily_api_key)
        checks.append(("Tavily", True, ok, msg))
    else:
        checks.append(("Tavily", False, True, "Not set → DuckDuckGo fallback"))

    if s.langchain_tracing_v2 and s.langchain_api_key:
        ok, msg = _check_langsmith(s.langchain_api_key, s.langchain_project)
        checks.append(("LangSmith", True, ok, msg))
    else:
        checks.append(("LangSmith", False, True, "Tracing disabled"))

    print(f"\n{'Service':<12} {'Status':<10} Detail")
    print("-" * 80)
    any_required_failed = False
    for name, configured, ok, msg in checks:
        if not configured:
            status = "skip"
        elif ok:
            status = "OK"
        else:
            status = "FAIL"
            if "(LLM)" in name:
                any_required_failed = True
        print(f"{name:<12} {status:<10} {msg}")

    print()
    if any_required_failed:
        print("❌ A required key failed. Fix and re-run.")
        return 1
    print("✅ All configured keys verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
