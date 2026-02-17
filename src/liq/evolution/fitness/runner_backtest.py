"""Backtest-based fitness evaluation via liq-runner."""

from __future__ import annotations

from typing import Any


class BacktestFitnessEvaluator:
    """Evaluates GP programs using backtested trading performance.

    Runs the top programs through a backtest engine and scores them
    on metrics like Sharpe ratio, Sortino ratio, or total return.
    """

    def evaluate(self, programs: list[Any], context: Any) -> list[Any]:
        """Evaluate programs via backtesting.

        Args:
            programs: GP programs to evaluate.
            context: Evaluation context with market data.

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
