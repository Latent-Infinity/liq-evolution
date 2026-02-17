"""Label-based fitness evaluation (f1, precision@k, accuracy)."""

from __future__ import annotations

from typing import Any


class LabelFitnessEvaluator:
    """Evaluates GP programs using label-based classification metrics.

    Computes fitness by comparing program outputs to labelled data
    using metrics like F1, precision@k, and accuracy.
    """

    def evaluate(self, programs: list[Any], context: Any) -> list[Any]:
        """Evaluate programs against labelled data.

        Args:
            programs: GP programs to evaluate.
            context: Evaluation context with features and labels.

        Returns:
            Fitness scores for each program.
        """
        raise NotImplementedError

    def evaluate_fitness(self, programs: list[Any], context: Any) -> list[Any]:
        """FitnessStageEvaluator protocol method.

        Args:
            programs: GP programs to evaluate.
            context: Evaluation context.

        Returns:
            Fitness results for each program.
        """
        raise NotImplementedError
