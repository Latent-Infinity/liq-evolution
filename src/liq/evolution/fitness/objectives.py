"""Objective function wiring for the fitness pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from liq.evolution.config import EvolutionRunConfig, FitnessStageConfig
from liq.evolution.fitness.label_metrics import LabelFitnessEvaluator
from liq.evolution.fitness.runner_backtest import BacktestFitnessEvaluator
from liq.evolution.fitness.two_stage import TwoStageFitnessEvaluator


def wire_objectives(
    config: FitnessStageConfig,
    *,
    backtest_fn: Callable[[Any], Sequence[dict[str, Any]]] | None = None,
    run_config: EvolutionRunConfig | None = None,
) -> LabelFitnessEvaluator | TwoStageFitnessEvaluator:
    """Wire up objective functions based on fitness stage configuration.

    Args:
        config: Fitness stage configuration.
        backtest_fn: Optional backtest runner callable. Required when
            ``config.use_backtest`` is ``True``.

    Returns:
        Configured evaluator (single-stage or two-stage).

    Raises:
        ValueError: If ``use_backtest=True`` but no ``backtest_fn``
            is provided.
    """
    stage_a = LabelFitnessEvaluator(
        metric=config.label_metric,
        top_k=config.label_top_k,
    )

    if not config.use_backtest:
        return stage_a

    if backtest_fn is None:
        raise ValueError(
            "backtest_fn is required when use_backtest=True. "
            "Pass a callable that accepts a strategy and returns fold results."
        )

    stage_b = BacktestFitnessEvaluator(
        backtest_runner=backtest_fn,
        metric=config.backtest_metric,
    )

    return TwoStageFitnessEvaluator(
        stage_a=stage_a,
        stage_b=stage_b,
        top_k=config.backtest_top_n,
        stage_b_candidate_budget=(
            run_config.stage_b_candidate_budget if run_config is not None else None
        ),
        stage_b_min_candidates=(
            run_config.stage_b_min_candidates if run_config is not None else 1
        ),
        stage_a_threshold=(
            run_config.stage_a_threshold if run_config is not None else None
        ),
    )
