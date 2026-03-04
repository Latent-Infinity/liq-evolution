"""Metric-resolution tests for StrategyEvaluator."""

from __future__ import annotations

from collections.abc import Mapping

from liq.datasets.walk_forward import WalkForwardSplit
from liq.evolution.fitness.strategy_evaluator import StrategyEvaluator, _resolve_metric
from liq.sim.fx_eval import turnover_from_positions


def test_resolve_metric_prefers_symbolic_aliases() -> None:
    metrics = {
        "cagr_ratio": 0.12,
        "max_dd": 0.41,
        "sharpe": 1.0,
    }
    derived: Mapping[str, float] = {
        "cagr": 0.01,
        "max_drawdown": 0.99,
        "sharpe": 0.5,
    }

    assert (
        _resolve_metric(
            objective_name="cagr",
            metrics=metrics,
            derived=derived,
        )
        == 0.12
    )
    assert (
        _resolve_metric(
            objective_name="max_drawdown",
            metrics=metrics,
            derived=derived,
        )
        == 0.41
    )
    assert (
        _resolve_metric(
            objective_name="sharpe",
            metrics=metrics,
            derived=derived,
        )
        == 1.0
    )


def test_resolve_metric_falls_back_to_derived_scores() -> None:
    metrics: dict[str, float] = {}
    derived = {"capacity_proxy": 0.15}

    assert (
        _resolve_metric(
            objective_name="capacity_proxy",
            metrics=metrics,
            derived=derived,
        )
        == 0.15
    )


def test_turnover_objective_uses_position_churn_metric() -> None:
    evaluator = StrategyEvaluator(
        backtest_runner=lambda _strategy, _context, _split: {},  # noqa: ARG005
        splits=[
            WalkForwardSplit(
                train=slice(0, 1),
                validate=slice(1, 2),
                test=slice(2, 3),
                slice_id="time_window:single",
            )
        ],
        objectives=("turnover",),
    )

    payload = {
        "metrics": {},
        "traces": {
            "position_trace": [1.0, 4.0, 2.0],
            "equity_curve": [100.0, 100.0, 100.0],
            "pnl_trace": [1.0, -1.0, 0.5],
        },
    }

    metrics = evaluator._extract_stage_metrics(payload)
    assert metrics["turnover"] == turnover_from_positions([1.0, 4.0, 2.0])
