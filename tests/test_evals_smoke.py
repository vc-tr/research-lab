from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from evals.judge import EvalScore, score_report
from evals.runner import load_questions, run_eval, write_report

FAKE_REPORT = """# Self-Attention Scaling in Transformers

## Introduction
Self-attention is the core mechanism in transformer architectures [1].

## Quadratic Complexity
The standard self-attention has O(n²) time and memory complexity with respect to
sequence length [1]. This becomes prohibitive for long sequences.

## Efficiency Approaches

### Flash Attention
FlashAttention reduces memory from O(n²) to O(n) using tiling and kernel fusion [2].

### Sparse Attention
Sparse patterns like Longformer use local + global attention for O(n) scaling [3].

### Linear Attention
Performers and other linear attention methods approximate softmax for O(n) complexity [4].

## References
[1] Vaswani et al., "Attention Is All You Need", NeurIPS 2017
[2] Dao et al., "FlashAttention", NeurIPS 2022
[3] Beltagy et al., "Longformer", arXiv 2020
[4] Choromanski et al., "Rethinking Attention with Performers", ICLR 2021
"""


def _fake_llm_client(prompt: str) -> tuple[str, int]:
    return json.dumps({"factual_accuracy": 4, "citation_correctness": 5, "completeness": 4}), 100


def _fake_report_generator(question: str, llm_client: Any = None) -> str:
    return FAKE_REPORT


class TestJudge:
    def test_score_report_returns_valid_scores(self) -> None:
        score = score_report(
            question_id="q1",
            question="How does self-attention scale?",
            report=FAKE_REPORT,
            expected_topics=["quadratic", "sparse attention", "flash attention"],
            rubric_notes="Should mention O(n²).",
            llm_client=_fake_llm_client,
        )
        assert isinstance(score, EvalScore)
        assert 1 <= score.factual_accuracy <= 5
        assert 1 <= score.citation_correctness <= 5
        assert 1 <= score.completeness <= 5
        assert score.error is None

    def test_score_report_handles_parse_error(self) -> None:
        def bad_llm(prompt: str) -> tuple[str, int]:
            return "not json", 50

        score = score_report(
            question_id="q1",
            question="test",
            report="test",
            expected_topics=[],
            rubric_notes="",
            llm_client=bad_llm,
        )
        assert score.error is not None
        assert score.factual_accuracy == 1


class TestRunner:
    def test_load_questions(self) -> None:
        questions = load_questions(Path("evals/questions.yaml"))
        assert len(questions) == 10
        assert all("question" in q for q in questions)
        assert all("expected_topics" in q for q in questions)

    def test_smoke_eval_completes(self) -> None:
        scores = run_eval(
            Path("evals/questions.yaml"),
            smoke=True,
            llm_client=_fake_llm_client,
            report_generator=_fake_report_generator,
        )
        assert len(scores) == 1
        assert scores[0].question_id == "q1"

    def test_write_report_creates_file(self) -> None:
        scores = [
            EvalScore(
                question_id="q1",
                factual_accuracy=4,
                citation_correctness=5,
                completeness=4,
                latency_seconds=1.5,
                total_tokens=100,
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_report(scores, Path(tmpdir))
            assert path.exists()
            content = path.read_text()
            assert "Eval Report" in content
            assert "q1" in content
