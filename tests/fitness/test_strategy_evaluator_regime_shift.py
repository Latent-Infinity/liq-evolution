"""Stage-6 synthetic regime-shift stress tests for StrategyEvaluator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from liq.datasets.walk_forward import WalkForwardSplit
from liq.evolution.fitness.evaluation_schema import METADATA_KEY_CONSTRAINT_VIOLATIONS
from liq.evolution.fitness.strategy_evaluator import StrategyEvaluator
from liq.evolution.validation import ConstraintPolicy
from liq.gp.program.ast import TerminalNode
from liq.gp.types import Series


def _splits() -> list[WalkForwardSplit]:
    return [
        WalkForwardSplit(
            train=slice(0, 8),
            validate=slice(8, 12),
            test=slice(12, 16),
            slice_id="time_window:split_0",
        ),
        WalkForwardSplit(
            train=slice(16, 24),
            validate=slice(24, 28),
            test=slice(28, 32),
            slice_id="time_window:split_1",
        ),
    ]


def _regime_shift_runner(
    strategy: Any,
    context: Mapping[str, Any],
    split: WalkForwardSplit,
) -> Mapping[str, Any]:
    del context
    program = getattr(strategy, "_program", None)
    name = str(getattr(program, "name", ""))
    split_index = 1 if split.slice_id.endswith("split_1") else 0

    if name == "overfit":
        train = {
            "cagr": 0.80 - split_index * 0.10,
            "turnover": 0.85,
            "max_drawdown": 0.45,
            "regime_coverage": 0.20,
        }
        validate = {
            "cagr": 0.05 - split_index * 0.02,
            "turnover": 0.90,
            "max_drawdown": 0.55,
            "regime_coverage": 0.15,
        }
    else:
        train = {
            "cagr": 0.30 - split_index * 0.02,
            "turnover": 0.20,
            "max_drawdown": 0.18,
            "regime_coverage": 0.80,
        }
        validate = {
            "cagr": 0.22 - split_index * 0.03,
            "turnover": 0.24,
            "max_drawdown": 0.19,
            "regime_coverage": 0.78,
        }

    return {
        "train": {"metrics": train},
        "validate": {"metrics": validate},
        "test": {"metrics": validate},
    }


def test_synthetic_regime_shift_filters_overfit_like_candidate() -> None:
    evaluator = StrategyEvaluator(
        backtest_runner=_regime_shift_runner,
        splits=_splits(),
        objectives=("cagr",),
        split_weights={"train": 0.2, "validate": 0.8},
        constraint_policy=ConstraintPolicy(
            robustness_rollout="standard",
            regime_coverage_floor=0.60,
            turnover_cap=0.40,
            drawdown_cap=0.25,
        ),
    )
    overfit = TerminalNode(name="overfit", output_type=Series)
    robust = TerminalNode(name="robust", output_type=Series)
    results = evaluator.evaluate([overfit, robust], context={"labels": [0.0, 1.0]})

    overfit_result, robust_result = results
    overfit_constraints = overfit_result.metadata[METADATA_KEY_CONSTRAINT_VIOLATIONS]
    robust_constraints = robust_result.metadata[METADATA_KEY_CONSTRAINT_VIOLATIONS]
    assert overfit_result.objectives[0] < robust_result.objectives[0]
    assert any(":robustness:" in key for key in overfit_constraints)
    assert not any(":robustness:" in key for key in robust_constraints)
    assert all(value > 0.0 for value in overfit_constraints.values())


def test_synthetic_regime_shift_is_reproducible_and_reason_codes_stable() -> None:
    evaluator = StrategyEvaluator(
        backtest_runner=_regime_shift_runner,
        splits=_splits(),
        objectives=("cagr",),
        split_weights={"train": 0.2, "validate": 0.8},
        constraint_policy=ConstraintPolicy(
            robustness_rollout="standard",
            regime_coverage_floor=0.60,
            turnover_cap=0.40,
            drawdown_cap=0.25,
        ),
    )
    programs = [
        TerminalNode(name="overfit", output_type=Series),
        TerminalNode(name="robust", output_type=Series),
    ]
    first = evaluator.evaluate(programs, context={"labels": [0.0, 1.0]})
    second = evaluator.evaluate(programs, context={"labels": [0.0, 1.0]})

    assert [result.objectives for result in first] == [result.objectives for result in second]
    assert [
        result.metadata[METADATA_KEY_CONSTRAINT_VIOLATIONS] for result in first
    ] == [result.metadata[METADATA_KEY_CONSTRAINT_VIOLATIONS] for result in second]
