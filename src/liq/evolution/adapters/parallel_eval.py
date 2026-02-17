"""Parallel evaluation of GP programs."""

from __future__ import annotations

from typing import Any


class ParallelEvaluator:
    """Evaluates GP programs in parallel using configurable backends.

    Supports sequential and Ray-based parallel evaluation.
    """

    def evaluate_batch(self, programs: list[Any], context: Any) -> list[Any]:
        """Evaluate a batch of programs, potentially in parallel.

        Args:
            programs: GP programs to evaluate.
            context: Shared evaluation context.

        Returns:
            Fitness results for each program.
        """
        raise NotImplementedError
