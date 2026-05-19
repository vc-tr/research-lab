from __future__ import annotations

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def _on_step(event_type: str, agent: str, content: str) -> None:
    color = {
        "agent_step": "cyan",
        "tool_call": "yellow",
    }.get(event_type, "white")
    console.print(f"[{color}][{agent}][/{color}] {content[:200]}")


@click.command()
@click.argument("question")
@click.option("--model", default=None, help="Override default LLM model")
@click.option("--no-pinecone", is_flag=True, help="Use in-memory vector store")
@click.option(
    "--max-subtasks",
    default=5,
    type=int,
    help="Number of parallel researcher agents (lower = less token usage)",
)
def main(question: str, model: str | None, no_pinecone: bool, max_subtasks: int) -> None:
    """Run a research query and produce a cited Markdown report."""
    from research_lab.config import get_settings
    from research_lab.crew import run_research

    console.print(Panel(f"[bold]Research Question:[/bold] {question}", style="blue"))

    settings = get_settings()
    if model:
        settings.default_model = model
    if no_pinecone:
        settings.use_pinecone = False

    with console.status("[bold green]Researching..."):
        report = run_research(
            question,
            settings=settings,
            step_callback=_on_step,
            max_subtasks=max_subtasks,
        )

    console.print()
    console.print(Markdown(report))


if __name__ == "__main__":
    main()
