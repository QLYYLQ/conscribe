"""LLM judge evaluator — uses cross-registry wiring to reference the LLM layer."""
from __future__ import annotations

from mock_framework.evaluators.exact_match import BaseEvaluator


class LLMJudgeEvaluator(BaseEvaluator):
    """Uses an LLM to judge result quality.

    Args:
        llm_provider: Which LLM provider to use for judging.
        rubric: Evaluation rubric prompt.
    """

    # Cross-registry wiring: references the mock_llm registry
    __wiring__ = {"llm_provider": "mock_llm"}

    def __init__(
        self,
        *,
        llm_provider: str = "openai",
        rubric: str = "Rate the quality from 0 to 1.",
    ):
        self.llm_provider = llm_provider
        self.rubric = rubric

    def evaluate(self, result: str) -> float:
        return 0.85
