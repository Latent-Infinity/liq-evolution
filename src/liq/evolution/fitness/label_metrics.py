"""Label-based fitness evaluation (F1, precision@k, accuracy)."""

from __future__ import annotations

import numpy as np

from liq.evolution.errors import FitnessError
from liq.gp.program.ast import Program
from liq.gp.program.eval import evaluate
from liq.gp.types import FitnessResult


class LabelFitnessEvaluator:
    """Evaluates GP programs using label-based classification metrics.

    Computes fitness by comparing program outputs to labelled data
    using metrics like F1, precision@k, and accuracy.
    """

    def __init__(
        self,
        metric: str = "f1",
        top_k: float = 0.1,
        turnover_weight: float = 0.0,
    ) -> None:
        """Initialize evaluator with metric configuration.

        Args:
            metric: Classification metric to use (f1, precision_at_k, accuracy).
            top_k: Fraction of top predictions to consider, in (0, 1].
            turnover_weight: Penalty weight for high-churn signals (>= 0).

        Raises:
            ValueError: If metric is unknown, top_k out of range,
                or turnover_weight is negative.
        """
        if metric not in ("f1", "precision_at_k", "accuracy"):
            raise ValueError(f"Unknown metric: {metric}")
        if not 0.0 < top_k <= 1.0:
            raise ValueError("top_k must be in (0, 1]")
        if turnover_weight < 0.0:
            raise ValueError("turnover_weight must be >= 0")
        self._metric = metric
        self._top_k = top_k
        self._turnover_weight = turnover_weight

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        """Evaluate programs against labelled data.

        Args:
            programs: GP programs to evaluate.
            context: Evaluation context with features and labels.

        Returns:
            Fitness results for each program.

        Raises:
            FitnessError: If *context* does not contain a ``"labels"`` key.
        """
        if "labels" not in context:
            raise FitnessError("Context must contain 'labels' key")
        labels = context["labels"]
        results: list[FitnessResult] = []
        for program in programs:
            result = self._evaluate_single(program, context, labels)
            results.append(result)
        return results

    def evaluate_fitness(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        """Backward-compatible alias for :meth:`evaluate`.

        Args:
            programs: GP programs to evaluate.
            context: Evaluation context.

        Returns:
            Fitness results for each program.
        """
        return self.evaluate(programs, context)

    # -------------------------------------------------------------- #
    #  Internal helpers
    # -------------------------------------------------------------- #

    def _evaluate_single(
        self,
        program: Program,
        context: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> FitnessResult:
        """Score a single program against *labels*."""
        scores = evaluate(program, context)

        # Handle all-NaN
        valid_mask = np.isfinite(scores)
        if not np.any(valid_mask):
            return FitnessResult(
                objectives=(0.0,),
                metadata={"metric": self._metric, "reason": "all_nan"},
            )

        # Replace NaN with -inf so they never rank as positive
        clean_scores = np.where(valid_mask, scores, -np.inf)

        # Compute metric value
        if self._metric == "precision_at_k":
            metric_value = self._precision_at_k(clean_scores, labels)
        elif self._metric == "f1":
            metric_value = self._f1(clean_scores, labels)
        else:  # accuracy
            metric_value = self._accuracy(clean_scores, labels)

        # Apply turnover penalty
        if self._turnover_weight > 0.0:
            turnover = self._compute_turnover(clean_scores)
            metric_value = metric_value * (1.0 - self._turnover_weight * turnover)
            metric_value = max(0.0, metric_value)

        return FitnessResult(
            objectives=(metric_value,),
            metadata={"metric": self._metric},
        )

    def _get_predictions(self, scores: np.ndarray) -> np.ndarray:
        """Convert scores to binary predictions using top_k threshold."""
        n = len(scores)
        k = max(1, int(n * self._top_k))
        threshold_idx = np.argsort(scores)[-k]
        threshold = scores[threshold_idx]
        return np.where(scores >= threshold, 1.0, 0.0)

    def _f1(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute F1 score from score-based predictions."""
        preds = self._get_predictions(scores)
        tp = float(np.sum((preds == 1.0) & (labels == 1.0)))
        fp = float(np.sum((preds == 1.0) & (labels == 0.0)))
        fn = float(np.sum((preds == 0.0) & (labels == 1.0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0.0:
            return 0.0
        return float(2.0 * precision * recall / (precision + recall))

    def _precision_at_k(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute precision among the top-k scored samples."""
        n = len(scores)
        k = max(1, int(n * self._top_k))
        top_k_idx = np.argsort(scores)[-k:]
        return float(np.mean(labels[top_k_idx]))

    def _accuracy(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute classification accuracy."""
        preds = self._get_predictions(scores)
        return float(np.mean(preds == labels))

    def _compute_turnover(self, scores: np.ndarray) -> float:
        """Compute signal turnover (fraction of prediction flips)."""
        binary = np.where(scores > np.median(scores), 1.0, 0.0)
        if len(binary) < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(binary))))
