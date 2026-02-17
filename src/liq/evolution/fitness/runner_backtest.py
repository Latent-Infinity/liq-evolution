"""Backtest-based fitness evaluation via liq-runner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl

from liq.evolution.adapters.signal_output import GPSignalOutput
from liq.evolution.errors import FitnessEvaluationError
from liq.gp.program.ast import Program
from liq.gp.program.eval import evaluate as gp_evaluate
from liq.gp.types import FitnessResult


class _ProgramStrategy:
    """Wraps a GP Program as a duck-type liq-runner Strategy."""

    def __init__(self, program: Program) -> None:
        self._program = program

    def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None:
        pass  # No-op: program is already evolved

    def predict(self, features: pl.DataFrame) -> GPSignalOutput:
        context = {col: features[col].to_numpy() for col in features.columns}
        scores_array = gp_evaluate(self._program, context)
        return GPSignalOutput(scores=pl.Series("scores", scores_array))


class BacktestFitnessEvaluator:
    """Evaluates GP programs using backtested trading performance.

    Uses dependency injection for the backtest runner function.
    """

    def __init__(
        self,
        backtest_runner: Callable[[Any], Sequence[dict[str, Any]]],
        metric: str = "sharpe_ratio",
    ) -> None:
        self._backtest_runner = backtest_runner
        self._metric = metric

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],  # noqa: ARG002
    ) -> list[FitnessResult]:
        results: list[FitnessResult] = []
        for program in programs:
            result = self._evaluate_single(program)
            results.append(result)
        return results

    def evaluate_fitness(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        return self.evaluate(programs, context)

    def _evaluate_single(self, program: Program) -> FitnessResult:
        strategy = _ProgramStrategy(program)

        try:
            fold_results = self._backtest_runner(strategy)
        except Exception as exc:
            raise FitnessEvaluationError(f"Backtest runner failed: {exc}") from exc

        if not fold_results:
            return FitnessResult(
                objectives=(0.0,),
                metadata={"metric": self._metric, "reason": "no_folds"},
            )

        metric_values: list[float] = []
        for fold in fold_results:
            metrics = fold.get("metrics", {})
            value = metrics.get(self._metric)
            if value is not None:
                metric_values.append(float(value))

        if not metric_values:
            return FitnessResult(
                objectives=(0.0,),
                metadata={"metric": self._metric, "reason": "metric_missing"},
            )

        avg_metric = float(np.mean(metric_values))
        return FitnessResult(
            objectives=(avg_metric,),
            metadata={
                "metric": self._metric,
                "n_folds": len(fold_results),
                "fold_values": metric_values,
            },
        )
