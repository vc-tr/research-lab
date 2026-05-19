FROM python:3.12-slim AS base

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY src/ src/
COPY evals/ evals/
COPY data/ data/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "research_lab.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
