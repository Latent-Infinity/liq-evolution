"""Tests for backtest-based fitness evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import pytest

from liq.evolution.adapters.signal_output import GPSignalOutput
from liq.evolution.errors import FitnessEvaluationError
from liq.evolution.fitness.runner_backtest import (
    BacktestFitnessEvaluator,
    _ProgramStrategy,
)
from liq.evolution.protocols import FitnessEvaluator
from liq.gp.program.ast import TerminalNode
from liq.gp.types import Series

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_program() -> TerminalNode:
    """Create a simple GP program (TerminalNode) for testing."""
    return TerminalNode("close", Series)


def _make_features() -> pl.DataFrame:
    """Create a small DataFrame with a 'close' column."""
    return pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})


def _mock_runner(folds: Sequence[dict[str, Any]]):
    """Create a backtest runner that returns predefined fold dicts."""

    def runner(strategy: Any) -> Sequence[dict[str, Any]]:
        return folds

    return runner


# ---------------------------------------------------------------------------
# _ProgramStrategy tests
# ---------------------------------------------------------------------------


class TestProgramStrategy:
    """Tests for the _ProgramStrategy wrapper."""

    def test_fit_is_noop(self) -> None:
        """fit() should succeed without errors or side effects."""
        program = _make_program()
        strategy = _ProgramStrategy(program)
        features = _make_features()
        # Should not raise; return value is None
        result = strategy.fit(features)
        assert result is None

    def test_fit_with_labels_is_noop(self) -> None:
        """fit() with labels should also be a no-op."""
        program = _make_program()
        strategy = _ProgramStrategy(program)
        features = _make_features()
        labels = pl.Series("label", [0, 1, 0, 1, 0])
        result = strategy.fit(features, labels)
        assert result is None

    def test_predict_returns_gp_signal_output(self) -> None:
        """predict() should return a GPSignalOutput instance."""
        program = _make_program()
        strategy = _ProgramStrategy(program)
        features = _make_features()
        output = strategy.predict(features)
        assert isinstance(output, GPSignalOutput)

    def test_predict_scores_length_matches_rows(self) -> None:
        """predict() scores length should match number of DataFrame rows."""
        program = _make_program()
        strategy = _ProgramStrategy(program)
        features = _make_features()
        output = strategy.predict(features)
        assert len(output.scores) == len(features)

    def test_predict_evaluates_program_correctly(self) -> None:
        """predict() should evaluate the GP program with DataFrame columns as context."""
        program = _make_program()
        strategy = _ProgramStrategy(program)
        features = _make_features()
        output = strategy.predict(features)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(output.scores.to_numpy(), expected)


# ---------------------------------------------------------------------------
# BacktestFitnessEvaluator construction tests
# ---------------------------------------------------------------------------


class TestBacktestFitnessEvaluatorConstruction:
    """Tests for BacktestFitnessEvaluator initialization."""

    def test_default_metric_is_sharpe_ratio(self) -> None:
        runner = _mock_runner([])
        evaluator = BacktestFitnessEvaluator(backtest_runner=runner)
        assert evaluator._metric == "sharpe_ratio"

    def test_custom_metric_stored(self) -> None:
        runner = _mock_runner([])
        evaluator = BacktestFitnessEvaluator(
            backtest_runner=runner, metric="total_return"
        )
        assert evaluator._metric == "total_return"


# ---------------------------------------------------------------------------
# BacktestFitnessEvaluator evaluation tests
# ---------------------------------------------------------------------------


class TestBacktestFitnessEvaluatorEvaluation:
    """Tests for BacktestFitnessEvaluator.evaluate()."""

    def test_single_program_single_fold(self) -> None:
        """Single program with one fold extracts metric correctly."""
        folds = [{"metrics": {"sharpe_ratio": 1.5, "total_return": 0.1}}]
        runner = _mock_runner(folds)
        evaluator = BacktestFitnessEvaluator(backtest_runner=runner)

        results = evaluator.evaluate([_make_program()], {})
        assert len(results) == 1
        assert results[0].objectives == pytest.approx((1.5,))

    def test_single_program_multiple_folds_averages(self) -> None:
        """Single program with multiple folds averages the metric."""
        folds = [
            {"metrics": {"sharpe_ratio": 1.0}},
            {"metrics": {"sharpe_ratio": 2.0}},
            {"metrics": {"sharpe_ratio": 3.0}},
        ]
        runner = _mock_runner(folds)
        evaluator = BacktestFitnessEvaluator(backtest_runner=runner)

        results = evaluator.evaluate([_make_program()], {})
        assert len(results) == 1
        assert results[0].objectives == pytest.approx((2.0,))

    def test_multiple_programs(self) -> None:
        """Multiple programs should each be evaluated."""
        folds = [{"metrics": {"sharpe_ratio": 1.0}}]
        runner = _mock_runner(folds)
        evaluator = BacktestFitnessEvaluator(backtest_runner=runner)

        programs = [_make_program(), _make_program(), _make_program()]
        results = evaluator.evaluate(programs, {})
        assert len(results) == 3

    def test_empty_programs_list(self) -> None:
        """Empty programs list should return empty results."""
        folds = [{"metrics": {"sharpe_ratio": 1.0}}]
        runner = _mock_runner(folds)
        evaluator = BacktestFitnessEvaluator(backtest_runner=runner)

        results = evaluator.evaluate([], {})
        assert results == []

    def test_runner_returns_empty_folds(self) -> None:
        """Runner returning no folds should produce FitnessResult(objectives=(0.0,))."""
        runner = _mock_runner([])
        evaluator = BacktestFitnessEvaluator(backtest_runner=runner)

        results = evaluator.evaluate([_make_program()], {})
        assert len(results) == 1
        assert results[0].objectives == (0.0,)

    def test_missing_metric_in_fold(self) -> None:
        """Fold without the target metric should produce FitnessResult(objectives=(0.0,))."""
        folds = [{"metrics": {"total_return": 0.1}}]
        runner = _mock_runner(folds)
        evaluator = BacktestFitnessEvaluator(backtest_runner=runner)

        results = evaluator.evaluate([_make_program()], {})
        assert len(results) == 1
        assert results[0].objectives == (0.0,)

    def test_runner_raises_exception(self) -> None:
        """Runner raising an exception should produce FitnessEvaluationError."""

        def bad_runner(strategy: Any) -> list[dict[str, Any]]:
            raise RuntimeError("connection failed")

        evaluator = BacktestFitnessEvaluator(backtest_runner=bad_runner)

        with pytest.raises(FitnessEvaluationError, match="Backtest runner failed"):
            evaluator.evaluate([_make_program()], {})

    def test_result_metadata_contains_metric_name(self) -> None:
        """Result metadata should contain the metric name."""
        folds = [{"metrics": {"sharpe_ratio": 1.5}}]
        runner = _mock_runner(folds)
        evaluator = BacktestFitnessEvaluator(backtest_runner=runner)

        results = evaluator.evaluate([_make_program()], {})
        assert results[0].metadata["metric"] == "sharpe_ratio"

    def test_result_metadata_contains_n_folds(self) -> None:
        """Result metadata should contain the number of folds."""
        folds = [
            {"metrics": {"sharpe_ratio": 1.0}},
            {"metrics": {"sharpe_ratio": 2.0}},
        ]
        runner = _mock_runner(folds)
        evaluator = BacktestFitnessEvaluator(backtest_runner=runner)

        results = evaluator.evaluate([_make_program()], {})
        assert results[0].metadata["n_folds"] == 2

    def test_result_metadata_contains_fold_values(self) -> None:
        """Result metadata should contain individual fold metric values."""
        folds = [
            {"metrics": {"sharpe_ratio": 1.0}},
            {"metrics": {"sharpe_ratio": 3.0}},
        ]
        runner = _mock_runner(folds)
        evaluator = BacktestFitnessEvaluator(backtest_runner=runner)

        results = evaluator.evaluate([_make_program()], {})
        assert results[0].metadata["fold_values"] == [1.0, 3.0]


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------


class TestBacktestFitnessEvaluatorProtocol:
    """Tests for protocol compliance."""

    def test_satisfies_fitness_evaluator_protocol(self) -> None:
        """BacktestFitnessEvaluator should satisfy the FitnessEvaluator protocol."""
        runner = _mock_runner([])
        evaluator = BacktestFitnessEvaluator(backtest_runner=runner)
        assert isinstance(evaluator, FitnessEvaluator)

    def test_evaluate_fitness_delegates_to_evaluate(self) -> None:
        """evaluate_fitness() should delegate to evaluate()."""
        folds = [{"metrics": {"sharpe_ratio": 2.5}}]
        runner = _mock_runner(folds)
        evaluator = BacktestFitnessEvaluator(backtest_runner=runner)

        programs = [_make_program()]
        result_eval = evaluator.evaluate(programs, {})
        result_fitness = evaluator.evaluate_fitness(programs, {})
        assert result_eval[0].objectives == result_fitness[0].objectives
