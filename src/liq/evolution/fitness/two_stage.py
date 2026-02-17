"""Two-stage fitness evaluation: fast screening followed by expensive refinement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from liq.gp.program.ast import Program
    from liq.gp.types import FitnessResult


class TwoStageFitnessEvaluator:
    """Two-stage fitness evaluator.

    Stage A evaluates all programs with a cheap metric (e.g. label-based).
    Stage B re-evaluates the top *top_k* programs with an expensive metric
    (e.g. backtest-based).  Programs outside the top K keep their Stage A
    results.

    When *stage_b* is ``None`` the evaluator acts as a passthrough for
    Stage A only.
    """

    def __init__(
        self,
        stage_a: Any,
        stage_b: Any | None = None,
        top_k: int = 10,
    ) -> None:
        """Initialize two-stage evaluator.

        Args:
            stage_a: Primary evaluator applied to all programs.
            stage_b: Optional secondary evaluator for top K programs.
            top_k: Number of top programs to promote to Stage B.

        Raises:
            ValueError: If *top_k* is less than 1.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._stage_a = stage_a
        self._stage_b = stage_b
        self._top_k = top_k

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        """Evaluate programs through one or two stages.

        Args:
            programs: GP programs to evaluate.
            context: Evaluation context (features, labels, etc.).

        Returns:
            Fitness results for each program, in input order.
        """
        # Stage A: evaluate all programs
        a_results = self._stage_a.evaluate(programs, context)

        if self._stage_b is None or len(programs) == 0:
            return a_results

        # Select top K by first objective (descending)
        k = min(self._top_k, len(programs))
        indexed = sorted(
            range(len(a_results)),
            key=lambda i: a_results[i].objectives[0],
            reverse=True,
        )
        top_k_indices = indexed[:k]

        # Stage B: evaluate top K only
        top_k_programs = [programs[i] for i in top_k_indices]
        b_results = self._stage_b.evaluate(top_k_programs, context)

        # Merge: top K get Stage B results, rest keep Stage A
        final = list(a_results)
        for local_idx, global_idx in enumerate(top_k_indices):
            final[global_idx] = b_results[local_idx]

        return final

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
