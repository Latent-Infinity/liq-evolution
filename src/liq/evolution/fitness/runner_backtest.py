"""Backtest-based fitness evaluation via liq-runner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl

from liq.evolution.adapters.signal_output import GPSignalOutput
from liq.evolution.errors import FitnessEvaluationError
from liq.evolution.fitness.evaluation_schema import (
    METADATA_KEY_BEHAVIOR_DESCRIPTORS,
    METADATA_KEY_CONSTRAINT_VIOLATIONS,
    METADATA_KEY_PER_SPLIT_METRICS,
    METADATA_KEY_RAW_OBJECTIVES,
    METADATA_KEY_SLICE_SCORES,
)
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

    def _make_metadata(
        self,
        *,
        metric_value: float,
        n_folds: int,
        fold_values: list[float],
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Build metadata payload matching phase-0 contract."""
        fold_metrics = {
            f"fold:{idx}": {"metric": value} for idx, value in enumerate(fold_values)
        }
        metadata: dict[str, Any] = {
            "metric": self._metric,
            "n_folds": n_folds,
            "fold_values": fold_values,
            METADATA_KEY_PER_SPLIT_METRICS: {
                "all": {"metric": metric_value},
                **fold_metrics,
            },
            METADATA_KEY_RAW_OBJECTIVES: (metric_value,),
            METADATA_KEY_BEHAVIOR_DESCRIPTORS: {},
            METADATA_KEY_CONSTRAINT_VIOLATIONS: {},
            METADATA_KEY_SLICE_SCORES: {},
        }
        if reason is not None:
            metadata["reason"] = reason
        return metadata

    def _evaluate_single(self, program: Program) -> FitnessResult:
        strategy = _ProgramStrategy(program)

        try:
            fold_results = self._backtest_runner(strategy)
        except Exception as exc:
            raise FitnessEvaluationError(f"Backtest runner failed: {exc}") from exc

        if not fold_results:
            return FitnessResult(
                objectives=(0.0,),
                metadata=self._make_metadata(
                    metric_value=0.0,
                    n_folds=0,
                    fold_values=[],
                    reason="no_folds",
                ),
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
                metadata=self._make_metadata(
                    metric_value=0.0,
                    n_folds=len(fold_results),
                    fold_values=[],
                    reason="metric_missing",
                ),
            )

        avg_metric = float(np.mean(metric_values))
        return FitnessResult(
            objectives=(avg_metric,),
            metadata=self._make_metadata(
                metric_value=avg_metric,
                n_folds=len(fold_results),
                fold_values=metric_values,
            ),
        )
