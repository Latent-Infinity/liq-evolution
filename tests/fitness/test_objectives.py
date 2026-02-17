"""Tests for objective wiring."""

from __future__ import annotations

import pytest

from liq.evolution.config import FitnessStageConfig
from liq.evolution.fitness.label_metrics import LabelFitnessEvaluator
from liq.evolution.fitness.objectives import wire_objectives
from liq.evolution.fitness.two_stage import TwoStageFitnessEvaluator


class TestWireObjectives:
    def test_default_config_returns_f1_evaluator(self) -> None:
        config = FitnessStageConfig()
        evaluator = wire_objectives(config)
        assert isinstance(evaluator, LabelFitnessEvaluator)

    def test_precision_at_k_config(self) -> None:
        config = FitnessStageConfig(label_metric="precision_at_k", label_top_k=0.2)
        evaluator = wire_objectives(config)
        assert isinstance(evaluator, LabelFitnessEvaluator)
        # Verify it uses the right metric by checking internal state
        assert evaluator._metric == "precision_at_k"
        assert evaluator._top_k == 0.2

    def test_accuracy_config(self) -> None:
        config = FitnessStageConfig(label_metric="accuracy")
        evaluator = wire_objectives(config)
        assert isinstance(evaluator, LabelFitnessEvaluator)
        assert evaluator._metric == "accuracy"

    def test_backtest_without_fn_raises(self) -> None:
        config = FitnessStageConfig(use_backtest=True)
        with pytest.raises(ValueError, match="backtest_fn.*required"):
            wire_objectives(config)

    def test_label_top_k_passed_through(self) -> None:
        config = FitnessStageConfig(label_top_k=0.3)
        evaluator = wire_objectives(config)
        assert evaluator._top_k == 0.3

    def test_backtest_with_fn_returns_two_stage(self) -> None:
        config = FitnessStageConfig(use_backtest=True, backtest_top_n=5)
        mock_fn = lambda strategy: [{"metrics": {"sharpe_ratio": 1.0}}]
        evaluator = wire_objectives(config, backtest_fn=mock_fn)
        assert isinstance(evaluator, TwoStageFitnessEvaluator)

    def test_backtest_two_stage_uses_correct_metric(self) -> None:
        config = FitnessStageConfig(
            use_backtest=True,
            backtest_metric="sortino_ratio",
        )
        mock_fn = lambda strategy: []
        evaluator = wire_objectives(config, backtest_fn=mock_fn)
        assert isinstance(evaluator, TwoStageFitnessEvaluator)
        # Stage B should use the configured metric
        assert evaluator._stage_b._metric == "sortino_ratio"

    def test_backtest_two_stage_uses_correct_top_n(self) -> None:
        config = FitnessStageConfig(use_backtest=True, backtest_top_n=7)
        mock_fn = lambda strategy: []
        evaluator = wire_objectives(config, backtest_fn=mock_fn)
        assert isinstance(evaluator, TwoStageFitnessEvaluator)
        assert evaluator._top_k == 7
