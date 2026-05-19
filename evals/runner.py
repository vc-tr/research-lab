from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from evals.judge import EvalScore, score_report


def load_questions(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _generate_report(
    question: str,
    llm_client: Any = None,
) -> str:
    from research_lab.config import get_settings
    from research_lab.crew import run_research

    settings = get_settings()
    return run_research(question, settings=settings)


def run_eval(
    questions_path: Path,
    smoke: bool = False,
    llm_client: Any = None,
    report_generator: Any = None,
) -> list[EvalScore]:
    questions = load_questions(questions_path)
    if smoke:
        questions = questions[:1]

    gen = report_generator or _generate_report
    scores: list[EvalScore] = []

    for q in questions:
        print(f"  Evaluating: {q['id']} — {q['question'][:60]}...")
        start = time.monotonic()
        report = gen(q["question"], llm_client=llm_client)
        gen_time = time.monotonic() - start

        score = score_report(
            question_id=q["id"],
            question=q["question"],
            report=report,
            expected_topics=q["expected_topics"],
            rubric_notes=q["rubric_notes"],
            llm_client=llm_client,
        )
        score.latency_seconds = gen_time + score.latency_seconds
        scores.append(score)

    return scores


def write_report(scores: list[EvalScore], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Eval Report",
        f"\nRun at: {datetime.now(UTC).isoformat()}",
        f"\nQuestions evaluated: {len(scores)}",
        "",
        "| ID | Accuracy | Citations | Completeness | Latency (s) | Tokens |",
        "|:---|:--------:|:---------:|:------------:|:-----------:|:------:|",
    ]

    for s in scores:
        lines.append(
            f"| {s.question_id} | {s.factual_accuracy} | {s.citation_correctness} "
            f"| {s.completeness} | {s.latency_seconds:.1f} | {s.total_tokens} |"
        )

    avg_acc = sum(s.factual_accuracy for s in scores) / max(len(scores), 1)
    avg_cit = sum(s.citation_correctness for s in scores) / max(len(scores), 1)
    avg_comp = sum(s.completeness for s in scores) / max(len(scores), 1)
    lines.append(
        f"\n**Averages**: Accuracy={avg_acc:.1f}, Citations={avg_cit:.1f}, "
        f"Completeness={avg_comp:.1f}"
    )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research eval suite")
    parser.add_argument(
        "--questions", default="evals/questions.yaml", help="Path to questions YAML"
    )
    parser.add_argument("--smoke", action="store_true", help="Run single-question smoke test")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) if args.output else Path(f"evals/reports/{timestamp}")

    print(f"Running eval ({'smoke' if args.smoke else 'full'})...")
    scores = run_eval(Path(args.questions), smoke=args.smoke)

    report_path = write_report(scores, output_dir)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
