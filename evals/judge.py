from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

GRADING_PROMPT = (  # noqa: E501
    "You are an expert research evaluator. "
    "Score the following research report on three dimensions.\n\n"
    "## Research Question\n{question}\n\n"
    "## Expected Topics\n{expected_topics}\n\n"
    "## Rubric Notes\n{rubric_notes}\n\n"
    "## Report to Evaluate\n{report}\n\n"
    "## Scoring Instructions\n"
    "Score each dimension from 1 to 5:\n"
    "- **factual_accuracy**: Are claims factually correct? "
    "(1=mostly wrong, 5=all correct)\n"
    "- **citation_correctness**: Are sources cited properly? "
    "(1=no citations, 5=all claims cited)\n"
    "- **completeness**: Does the report cover expected topics? "
    "(1=misses most, 5=comprehensive)\n\n"
    "Return ONLY a JSON object with these three keys and integer "
    "values 1-5. No other text.\n"
    'Example: {{"factual_accuracy": 4, "citation_correctness": 3, '
    '"completeness": 5}}'
)


@dataclass
class EvalScore:
    question_id: str
    factual_accuracy: int
    citation_correctness: int
    completeness: int
    latency_seconds: float = 0.0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    error: str | None = None


def score_report(
    question_id: str,
    question: str,
    report: str,
    expected_topics: list[str],
    rubric_notes: str,
    llm_client: Any = None,
) -> EvalScore:
    prompt = GRADING_PROMPT.format(
        question=question,
        expected_topics=", ".join(expected_topics),
        rubric_notes=rubric_notes,
        report=report,
    )

    start = time.monotonic()

    if llm_client is not None:
        response_text, tokens = llm_client(prompt)
    else:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text  # type: ignore[union-attr]
        tokens = response.usage.input_tokens + response.usage.output_tokens

    elapsed = time.monotonic() - start

    try:
        scores = json.loads(response_text)
        return EvalScore(
            question_id=question_id,
            factual_accuracy=int(scores["factual_accuracy"]),
            citation_correctness=int(scores["citation_correctness"]),
            completeness=int(scores["completeness"]),
            latency_seconds=elapsed,
            total_tokens=tokens,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return EvalScore(
            question_id=question_id,
            factual_accuracy=1,
            citation_correctness=1,
            completeness=1,
            latency_seconds=elapsed,
            error=f"Failed to parse judge response: {e}",
        )
