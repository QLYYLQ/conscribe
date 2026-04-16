"""Exact match evaluator — simple evaluator, no wiring."""
from __future__ import annotations

from mock_framework.layers import EvaluatorRegistrar


class BaseEvaluator(metaclass=EvaluatorRegistrar.Meta):
    __abstract__ = True

    def evaluate(self, result: str) -> float:
        return 0.0


class ExactMatchEvaluator(BaseEvaluator):
    """Evaluates via exact string match.

    Args:
        case_sensitive: Whether comparison is case-sensitive.
    """

    def __init__(self, *, case_sensitive: bool = True):
        self.case_sensitive = case_sensitive

    def evaluate(self, result: str) -> float:
        return 1.0
