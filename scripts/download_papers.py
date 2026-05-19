#!/usr/bin/env python3
"""Download arXiv papers for the research corpus."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

PAPERS: dict[str, str] = {
    "2210.03629": "ReAct: Synergizing Reasoning and Acting in Language Models",
    "2302.04761": "Toolformer: Language Models Can Teach Themselves to Use Tools",
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "2312.10997": "RAG Survey: Retrieval-Augmented Generation for AI-Generated Content",
    "2310.11511": "Self-RAG: Learning to Retrieve, Generate, and Critique",
    "2309.15217": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
    "2310.06825": "DSPy: Compiling Declarative Language Model Calls",
    "2402.14207": "STORM: Synthesis of Topic Outlines through Retrieval and Multi-perspective QA",
    "2308.08155": "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
    "2303.17580": "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face",
    "2305.10601": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
    "2307.09288": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
    "2312.06648": "Mixtral of Experts",
    "2401.04088": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts LLM",
    "2305.14314": "QLoRA: Efficient Finetuning of Quantized Language Models",
    "2402.01680": "LLM Judges: A Survey on LLM-as-a-Judge",
    "2404.10667": "Many-Shot In-Context Learning",
    "2305.18290": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
    "2312.11805": "The Landscape of Emerging AI Agent Architectures",  # noqa: E501
    "2406.12832": "GraphRAG: From Local to Global with Graph-Based RAG",
}


def download_papers(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=60.0, follow_redirects=True)

    for arxiv_id, title in PAPERS.items():
        filename = f"{arxiv_id}.pdf"
        filepath = output_dir / filename
        if filepath.exists():
            print(f"  [skip] {filename} ({title})")
            continue

        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"  [download] {filename} — {title}")
        try:
            resp = client.get(url)
            resp.raise_for_status()
            filepath.write_bytes(resp.content)
        except httpx.HTTPError as e:
            print(f"  [error] {filename}: {e}", file=sys.stderr)

    client.close()


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/papers")
    print(f"Downloading {len(PAPERS)} papers to {target}/")
    download_papers(target)
    print("Done.")
